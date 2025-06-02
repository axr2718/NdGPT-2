from models.gpt2 import GPT2
from datasets.finewebedu_10bt import FineWebEdu10BT
from config import GPT2Config, TrainConfig
import torch
import torch.nn as nn
from experiment.train import train_gpt2
from utils.set_seed import set_seed

torch.set_float32_matmul_precision('high')

if '__main__' == __name__:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model_config = GPT2Config()
    train_config = TrainConfig()

    seed = train_config.seed
    set_seed(seed)

    dataset_dir = train_config.dataset_dir
    
    batch_size = train_config.batch_size
    seq_len = train_config.seq_len
    weight_decay = train_config.weight_decay
    learning_rate = train_config.max_lr

    train_loader = FineWebEdu10BT(batch_size=batch_size, seq_len=seq_len, dir_path=dataset_dir)
    val_loader = FineWebEdu10BT(batch_size=batch_size, seq_len=seq_len, dir_path=dataset_dir, split='val')

    model = GPT2(config=model_config)

    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            nodecay_params.append(param)
        else: decay_params.append(param)

    model_params = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    model = torch.compile(model)
    model.eval()
    model.to(device=device)

    optimizer = torch.optim.AdamW(model_params, lr=learning_rate, betas=(0.9, 0.95), fused=True)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_gpt2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        train_config=train_config
        )