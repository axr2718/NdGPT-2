from dataclasses import dataclass

@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768

@dataclass
class TrainConfig:
    max_lr: float = 6e-4 * 3
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.1

    max_batch_size: int = 524288
    B: int = 8
    T: int = 1024
    grad_steps: int = max_batch_size // (B * T)