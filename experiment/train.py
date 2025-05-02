from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models.gpt2 import GPTDataLoader
from config import TrainConfig
import math
import os
from .test import evaluate_gpt2, evaluate_hellaswag
from .generate import generate_text

def train_gpt2(model: nn.Module,
               trainloader: GPTDataLoader,
               valloader: GPTDataLoader,
               optimizer: torch.optim.Optimizer,
               train_config: TrainConfig
               ) -> nn.Module:
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ndgpt2-log.txt')
    with open(log_file, 'w') as file:
        pass

    warmup_steps = train_config.warmup_steps
    max_lr = train_config.max_lr
    max_steps = train_config.max_steps
    min_lr = train_config.min_lr
    grad_steps = train_config.grad_steps

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
    
    for step in range(3087, max_steps):
        last_step = (step == max_steps - 1)

        if (step % 250 ==0) or last_step:
            evaluate_gpt2(model=model,
                          optimizer=optimizer,
                          valloader=valloader,
                          step=step,
                          last_step=last_step,
                          log_file=log_file)
            
        if (step % 250 == 0 or last_step):
            evaluate_hellaswag(model=model, log_file=log_file, step=step)
            
        if ((step > 0 and step % 250 == 0) or last_step):
            generate_text(model=model)
        
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        for grad_step in range(grad_steps):
            x, y = trainloader.get_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss /= grad_steps
            total_loss += loss.detach()
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        print(f'Step {step} | Loss {total_loss.item():.4f} | lr {lr:.6f} | Norm {norm:.4f}')
        with open(log_file, 'a') as file:
            file.write(f'{step} train {total_loss.item():.6f}\n')

    return model

def train_vision(model: nn.Module,
                 train_dataset: Dataset,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 batch_size: int) -> nn.Module:
    
    model.train()
    
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            norm = clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
        
        epoch_loss /= len(train_dataset)
        print(f'Epoch: {epoch + 1} | Loss: {epoch_loss:.4f}')


    return model