# Algorithm taken from https://arxiv.org/abs/2306.11695
import torch
import torch.nn as nn
from ndlinear import NdLinear

def prune_linear_layer(layer: nn.Linear, X: torch.Tensor, sparsity: float):
    W = layer.weight.data 
    
    x_norm = X.norm(p=2, dim=0)       
    
    metric = W.abs() * x_norm.unsqueeze(0)
    
    C_out, C_in = W.shape
    k = int(C_in * sparsity)
    if k == 0:
        return

    _, idx = metric.sort(dim=1)       
    to_prune = idx[:, :k]             
    W.scatter_(1, to_prune, 0.0)


def prune_ndlinear_layer(layer: NdLinear, X: torch.Tensor, sparsity: float):
    dims = X.shape[1:]

    for mode, (Di, Hi) in enumerate(zip(layer.input_dims, layer.hidden_size)):
        Wi = layer.weights[mode].data    # (Di, Hi)

        perm = [0] + [i+1 for i in range(len(dims)) if i != mode] + [mode+1]
        Xp = X.permute(*perm)
        Xm = Xp.reshape(-1, Di) 

        
        x_norm = Xm.norm(p=2, dim=0)
        
        metric = Wi.abs() * x_norm.unsqueeze(1)

        
        k = int(Di * sparsity)
        if k == 0:
            continue
        
        mT = metric.t()                   
        Wt = Wi.t()                       
        _, idx = mT.sort(dim=1)          
        to_prune = idx[:, :k]             
        Wt.scatter_(1, to_prune, 0.0)

        layer.weights[mode].data.copy_(Wt.t())


def prune_model(model: nn.Module,
                dataloader,
                sparsity: float,
                device: torch.device,
                n_calib_batches: int = 10):

    model.eval()
    activations = {}
    handles = []

    for name, module in model.named_modules():
        is_linear = isinstance(module, nn.Linear) and module is not getattr(model, "head", None)
        is_ndlinear = isinstance(module, NdLinear)
        if is_linear or is_ndlinear:
            activations[name] = []
            def make_hook(nm):
                def hook(mod, inp, out):
                    X = inp[0].detach()
                    if X.dim() > 2:
                        X = X.reshape(-1, X.size(-1))
                    activations[nm].append(X)
                return hook
            handles.append(module.register_forward_hook(make_hook(name)))

    for _ in range(n_calib_batches):
        x, _ = dataloader.get_batch()
        x = x.to(device)
        with torch.no_grad():
            _ = model(x)

    for h in handles:
        h.remove()

    for name, module in model.named_modules():
        if name not in activations:
            continue
        X_cat = torch.cat(activations[name], dim=0)
        if isinstance(module, nn.Linear):
            prune_linear_layer(module, X_cat, sparsity)
        elif isinstance(module, NdLinear):

            chunk = activations[name][0]
            in_dims = chunk.shape[1:]
            X_nd = X_cat.reshape(-1, *in_dims)
            prune_ndlinear_layer(module, X_nd, sparsity)

    return model
