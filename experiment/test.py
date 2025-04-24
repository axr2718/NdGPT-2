import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,
                     precision_score,
                     recall_score,
                     f1_score)

def test(model: nn.Module, test_dataset: Dataset, device: torch.device) -> dict:
    model.eval()

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