import torch
import torch.nn as nn
from datasets.finewebedu_10bt import FineWebEdu10BT
from config import TrainConfig
import math
import os
from .test import evaluate_gpt2, evaluate_hellaswag
from .generate import generate_text

def train_gpt2(
        model: nn.Module,
        train_loader: FineWebEdu10BT,
        val_loader: FineWebEdu10BT,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_config: TrainConfig
        ) -> nn.Module:

    warmup_steps = train_config.warmup_steps
    max_lr = train_config.max_lr
    max_steps = train_config.max_steps
    min_lr = train_config.min_lr
    grad_steps: int = train_config.max_batch_size // (train_config.batch_size * train_config.seq_len)
    start_step = train_config.start_step

    log_dir = train_config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{model.__class__.__name__}-log.txt')

    if not os.path.exists(log_path):
        with open(log_path, 'w') as file:
            pass

    checkpoint_dir = train_config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    
    def get_lr(it: int) -> float:
        if it < warmup_steps:
            return max_lr * (it + 1) / warmup_steps
        
        if it > max_steps:
            return min_lr
        
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)

        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)
    
    for step in range(start_step, max_steps):
        last_step = (step == max_steps - 1)

        if (step % 250 ==0) or last_step:
            val_loss = evaluate_gpt2(model=model, criterion=criterion, val_loader=val_loader)
            print(f'Validation Loss {val_loss:.4f}')

            with open(log_path, 'a') as file:
                file.write(f'{step} val {val_loss:.4f}\n')

            if (step > 0) and (step % 500 == 0 or last_step):
                train_config.start_step = step
                checkpoint_path = os.path.join(checkpoint_dir, f'{model.__class__.__name__}_{step:05d}.pt')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_config': train_config
                    }
                torch.save(checkpoint, checkpoint_path)
            
        if (step % 250 == 0 or last_step):
            num_correct, num_total = evaluate_hellaswag(model=model)
            acc = num_correct / num_total
            print(f'HellaSwag accuracy: {num_correct}/{num_total} = {acc:.4f}')

            with open(log_path, 'a') as file:
                file.write(f'{step} hella {acc:.4f}\n')
            
        if ((step > 0 and step % 250 == 0) or last_step):
            generate_text(model=model, seed=train_config.seed)
        
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for grad_step in range(grad_steps):
            x, y = train_loader.get_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            loss /= grad_steps
            total_loss += loss.detach()
            loss.backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        print(f'Step {step} | Loss {total_loss.item():.4f} | lr {lr:.6f} | Norm {norm:.4f}')

        with open(log_path, 'a') as file:
            file.write(f'{step} train {total_loss.item():.6f}\n')

    return model