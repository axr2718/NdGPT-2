from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

def train(model: nn.Module,
          train_dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          batch_size: int):
    
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