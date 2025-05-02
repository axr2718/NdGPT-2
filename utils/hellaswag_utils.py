import tiktoken
import torch
import torch.nn.functional as F
import os
import json
from typing import Iterator, Dict, Any, Tuple

def get_row(tokens: torch.LongTensor, mask: torch.LongTensor, logits: torch.Tensor) -> int:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()

    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)

    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    
    pred = avg_loss.argmin().item()

    return pred

def iterate_examples(split: str, dir_path: str = './data/hellaswag') -> Iterator[Dict[str, Any]]:
    with open(os.path.join(dir_path, f'hellaswag_{split}.jsonl'), 'r') as file:
        for line in file:
            example = json.loads(line)
            yield example

def render_example(example: dir) -> Tuple[Dict[str,Any], torch.LongTensor, torch.LongTensor, int]:
    encoding = tiktoken.get_encoding('gpt2')

    ctx = example['ctx']
    label = example['label']
    endings = example['endings']

    data =  {'label': label,
             'ctx_tokens': None,
             'ending_tokens': []}
    
    ctx_tokens = encoding.encode(ctx)
    data['ctx_tokens'] = ctx_tokens
    tok_rows = []
    mask_rows = []

    for end in endings:
        end_tokens = encoding.encode(' ' + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data['ending_tokens'].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

