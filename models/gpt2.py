import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from config import GPT2Config
import numpy as np
from typing import Tuple

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0

        self.attn = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        qkv = self.attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config, scale=True) -> nn.Module:
        super().__init__()
        self.config = config
        self.scale = scale

        self.transformer = nn.ModuleDict(dict(wte = nn.Embedding(config.vocab_size, config.embed_dim),
                                              wpe = nn.Embedding(config.block_size, config.embed_dim),
                                              h = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                                              ln_f = nn.LayerNorm(config.embed_dim),))
        
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.head.weight

        self.apply(self._init_params)

    def _init_params(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std=0.02

            # CHECK SCALE
            if self.scale:
                std *= (2 * self.config.num_layers)**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            else: torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets=None) -> torch.Tensor:
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot compute"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
class GPTDataLoader():
    def __init__(self, B: int, T: int, dir_path: str, split: str = 'train') -> None:
        self.B = B
        self.T = T

        assert split in {'train', 'val'}
        
        shards = os.listdir(dir_path)
        shards = [shard for shard in shards if split in shard]
        shards = sorted(shards)
        shards = [os.path.join(dir_path, shard) for shard in shards]
        self.shards = shards

        assert len(shards) > 0, "No shards found"

        self._reset()

    def _load_tokens(self, filename: str):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)

        return ptt
    
    def _reset(self):
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.position = self.B * self.T

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        buffer = self.tokens[self.position : self.position + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)

        self.position += B * T

        if self.position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.position = B * T 

        return x, y