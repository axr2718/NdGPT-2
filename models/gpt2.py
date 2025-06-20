import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPT2Config
from ndlinear import NdLinear

class Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads


        self.attn = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.size()

        qkv = self.attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        q = q.view(batch_size, seq_len, self.num_heads, dim // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, dim // self.num_heads).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, dim // self.num_heads).transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        x = self.proj(x)

        return x


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        self.embed_dim = config.embed_dim
        self.intermediate_size = config.intermediate_size
        self.filter_size = config.filter_size

        self.use_ndlinear = config.use_ndlinear

        if not self.use_ndlinear:
            self.fc1 = nn.Linear(self.embed_dim, self.intermediate_size)
            self.fc2 = nn.Linear(self.intermediate_size, self.embed_dim)
        else: 
            self.fc1 = NdLinear((self.embed_dim, 1), (self.intermediate_size, self.filter_size))
            self.fc2 = NdLinear((self.intermediate_size, self.filter_size), (self.embed_dim, 1))

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        if not self.use_ndlinear:
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
        else: 
            x = x.view(-1, x.size(-1))
            x = x.unsqueeze(-1)
            x = self.fc1(x)
            x = self.gelu(x)
            x = self.fc2(x)
            #x.squeeze(-1)
            return x.view(batch_size, seq_len, embed_dim)  
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        self.embed_dim = config.embed_dim

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> nn.Module:
        super().__init__()
        self.config = config

        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(self.vocab_size, self.embed_dim),
            'wpe': nn.Embedding(self.max_seq_len, self.embed_dim),
            'h': nn.ModuleList([Block(config) for _ in range(self.num_layers)]),
            'norm': nn.LayerNorm(self.embed_dim)
            })

        self.head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)

        self.transformer.wte.weight = self.head.weight

        self.apply(self._init_params)

    def _init_params(self, module):
        mean = 0.0
        std = 0.02
        residual_scale = (2 * self.num_layers) ** -0.5

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=mean, std=std)
        elif isinstance(module, Attention) or isinstance(module, MLP):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.normal_(submodule.weight, mean=mean, std=std*residual_scale)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        _, seq_len = input.size()
        assert seq_len <= self.max_seq_len, "Sequence length exceeded maximum sequence length."
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input.device)
        positional_embedding = self.transformer.wpe(positions)
        token_embedding = self.transformer.wte(input)
        x = positional_embedding + token_embedding

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.norm(x)
        logits = self.head(x)

        return logits