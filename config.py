from dataclasses import dataclass
import math

@dataclass
class GPT2Config:
    max_seq_len: int = 4096
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    intermediate_size: int = embed_dim * 4 // 2
    filter_size: int = 8

    use_ndlinear: bool = True

@dataclass
class TrainConfig:
    model_name: str = 'NdGPT2-96M'
    dataset_name: str = 'HuggingFaceFW/fineweb-edu'
    seed: int = 13
    max_lr: float = 1e-3
    min_lr: float = max_lr * 0.1
    max_steps: int = 20000
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_tokens: int = 4194304
    batch_size: int = 1
    num_workers: int = 2
    seq_len: int = 512 # 4096
    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    gradient_accumulation_steps: int = math.ceil(max_tokens / (batch_size * seq_len))
    use_ndlinear: bool = True
    use_ddp: bool = False

# GPT-2 parameters: 124475904
# NdGPT-2 parameters: 81980976