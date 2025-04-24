import torch.nn as nn
import torch

def parameter_count(model: nn.Module):
    num_params = 0
    for params in model.parameters():
        num_params += torch.numel(params)

    return num_params
        