import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from models.gpt2 import GPTDataLoader
import os
from utils.hellaswag_utils import get_row, iterate_examples, render_example
from sklearn.metrics import (accuracy_score,
                     precision_score,
                     recall_score,
                     f1_score)

@torch.inference_mode()
def evaluate_gpt2(model: nn.Module,
                  valloader: GPTDataLoader,
                  step: int,
                  last_step:int,
                  log_file: str) -> None:
    
    model.eval()
    device = next(model.parameters()).device
    valloader._reset()

    with torch.no_grad():
        val_loss = 0.0
        val_steps = 20

        for _ in range (val_steps):
            x, y = valloader.get_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(x, y)

            loss = loss / val_steps
            val_loss += loss.detach()

    print(f'Validation Loss {val_loss.item():.4f}')
    with open(log_file, 'a') as file:
        file.write(f'{step} val {val_loss.item():.4f}\n')

    if (step > 0) and (step % 500 == 0 or last_step):
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{step:05d}.pt')
        checkpoint = {'model': model.state_dict(),
                      'config': model.config,
                      'step': step,
                      'val_loss': val_loss.item()}
        torch.save(checkpoint, checkpoint_path)

@torch.inference_mode()
def evaluate_hellaswag(model: nn.Module, log_file: str, step: int):
    device = next(model.parameters()).device
    num_correct = 0
    num_total = 0

    for i, example in enumerate(iterate_examples(split='val')):
        
        _, tokens, mask, label = render_example(example=example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, loss = model(tokens)
            
            pred = get_row(tokens, mask, logits)
        
        num_total += 1
        num_correct += int(pred == label)

    acc = num_correct / num_total
    print(f'HellaSwag accuracy: {num_correct}/{num_total} = {acc:.4f}')
    with open(log_file, 'a') as file:
        file.write(f'{step} hella {acc:.4f}\n')


def test_vision(model: nn.Module, test_dataset: Dataset) -> dict:
    model.eval()
    device = next(model.parameters()).device

    testloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            loss = F.cross_entropy(y_hat, y)

            total_loss += loss.item() * x.size(0)

            _, predictions = torch.max(y_hat, 1)

            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    metrics = {
        'loss': total_loss / len(test_dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1}
    
    return metrics