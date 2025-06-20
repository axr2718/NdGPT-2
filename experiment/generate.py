import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# TODO
@torch.inference_mode()
def generate_text(model: nn.Module, tokenizer: AutoTokenizer):
    model.eval()
    device = next(model.parameters()).device

    return_sequences = 4
    max_length = 32

    tokens = tokenizer.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(return_sequences, 1)

    gen = tokens.to(device)
    while gen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(gen)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(input=probs, k=50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            col = torch.gather(topk_indices, -1, ix)
            gen = torch.cat((gen, col), dim=1)

    for i in range(return_sequences):
        tokens = gen[i, :max_length].tolist()
        decoded = tokenizer.decode(tokens)
        print(f'Sample {i}: {decoded}')