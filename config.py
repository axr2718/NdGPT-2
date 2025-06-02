from dataclasses import dataclass

@dataclass
class GPT2Config:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768

@dataclass
class TrainConfig:
    dataset_dir: str = './data/edu_fineweb10B'
    seed: int = 13
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 715
    max_steps: int = 19073
    weight_decay: float = 0.1

    max_batch_size: int = 524288
    batch_size: int = 8
    seq_len: int = 1024

    start_step: int = 0

    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'