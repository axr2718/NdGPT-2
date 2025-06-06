from dataclasses import dataclass

@dataclass
class GPT2Config:
    max_seq_len: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768

    use_ndlinear: bool = True
    h1: int = 528
    h2: int = 528


@dataclass
class TrainConfig:
    model_name: str = 'NdGPT2-82M'
    dataset_dir: str = './data/edu_fineweb10B'
    seed: int = 13

    epochs: int = 8
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1

    max_batch_size: int = 524288
    batch_size: int = 64
    seq_len: int = 1024

    start_step: int = 0

    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'

# GPT-2 parameters: 124475904
# NdGPT-2 parameters: 81980976