import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import TrainConfig
import os
from .test import evaluate_hellaswag
from .generate import generate_text
from utils.cosine_decay_with_warmup import cosine_decay_with_warmup
from transformers import AutoTokenizer
import time
import math

def train_gpt2(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        tokenizer: AutoTokenizer,
        train_config: TrainConfig,
        rank: int = 0,
        world_size: int = 0
        ) -> nn.Module:
    
    warmup_steps = train_config.warmup_steps
    max_lr = train_config.max_lr
    min_lr = train_config.min_lr
    max_steps = train_config.max_steps
    gradient_accumulation_steps = math.ceil(train_config.gradient_accumulation_steps // world_size)
    start_step = train_config.start_step
    optimization_step = train_config.optimization_step
    
    if rank == 0:
        last_step = (optimization_step == max_steps - 1)

        print(f'Backward pass at each {gradient_accumulation_steps} steps for {train_config.max_tokens} tokens, given {train_config.batch_size * train_config.seq_len * world_size} tokens per step')
        
        log_dir = train_config.log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{train_config.model_name}-log.txt')

        if not os.path.exists(log_path):
            with open(log_path, 'w') as file:
                pass

        checkpoint_dir = train_config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    device = next(model.parameters()).device

    optimizer.zero_grad() 
    total_loss = 0
    tokens_processed = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader, start=start_step):
        model.train()

        input_ids = batch['input_ids'].to(device)

        tokens_processed += input_ids.numel()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)
            logits = logits[:, :-1].contiguous()
            targets = input_ids[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss /= gradient_accumulation_steps

        total_loss += loss.detach()

        if (step + 1) % gradient_accumulation_steps == 0:
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr = cosine_decay_with_warmup(step=optimization_step, warmup_steps=warmup_steps, max_steps=max_steps, max_lr=max_lr,min_lr=min_lr)
        
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()
            optimizer.zero_grad()

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            total_time = end_time - start_time

            tokens_per_sec = tokens_processed / total_time
            
            if train_config.use_ddp:
                torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)

            if rank == 0:
                print(
                    f"Step: {optimization_step} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"LR: {lr:.4e} | "
                    f"Norm: {norm:.4f} | "
                    f"Tokens/Second: {tokens_per_sec * world_size:.2f} | "
                    f"Using NdLinear: {train_config.use_ndlinear}"
                    )
                
                with open(log_path, 'a') as file:
                    file.write(f'{optimization_step} train {total_loss.item():.6f}\n')
            
            #optimization_step += 1

                if (optimization_step + 1) % 10 == 0:
                    ckpt = {
                        'step': step + 1,
                        'optimization_step': optimization_step + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'dataloader': dataloader.state_dict(),
                    }

                    ckpt_dir = f'./checkpoints/models/{train_config.model_name}'
                    os.makedirs(ckpt_dir, exist_ok=True)
                    save_ckpt_path = os.path.join(ckpt_dir, f'{train_config.model_name}.pt')
                    torch.save(ckpt, save_ckpt_path)

                    model.eval()    
                    with torch.no_grad():
                        if (optimization_step > 0 or last_step):
                            generate_text(model=model, tokenizer=tokenizer)

                    hellaswag_accuracy = evaluate_hellaswag(model)
                    
                    print(f'HellaSwag accuracy: {hellaswag_accuracy * 100: .2f}')
                    with open(log_path, 'a') as file:
                        file.write(f'{optimization_step} hella {hellaswag_accuracy:.4f}\n')

            total_loss = 0
            tokens_processed = 0
            start_time = time.time()
            optimization_step += 1
        else:

            if train_config.use_ddp:
                with model.no_sync():
                    loss.backward()
            else: loss.backward()

            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

