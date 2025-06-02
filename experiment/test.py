import torch
import torch.nn as nn
from datasets.finewebedu_10bt import FineWebEdu10BT
from utils.hellaswag_utils import get_row, iterate_examples, render_example
from typing import Tuple

@torch.inference_mode()
def evaluate_gpt2(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: FineWebEdu10BT,
    ) -> float:
    
    model.eval()
    device = next(model.parameters()).device
    val_loader._reset()

    with torch.no_grad():
        val_loss = 0.0
        val_steps = 20

        for _ in range (val_steps):
            x, y = val_loader.get_batch()
            x, y = x.to(device), y.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss / val_steps
            val_loss += loss.detach()

    return val_loss.item()

@torch.inference_mode()
def evaluate_hellaswag(model: nn.Module) -> Tuple[int, int]:
    device = next(model.parameters()).device
    num_correct = 0
    num_total = 0

    for i, example in enumerate(iterate_examples(split='val')):
        
        _, tokens, mask, label = render_example(example=example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(tokens)
            
            pred = get_row(tokens, mask, logits)
        
        num_total += 1
        num_correct += int(pred == label)

    return num_correct, num_total