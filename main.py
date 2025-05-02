from models.gpt2 import GPTDataLoader
from models.ndgpt2 import NdGPT2
from config import GPT2Config, TrainConfig
import torch
from experiment.train import train_gpt2
from utils.set_seed import set_seed


torch.set_float32_matmul_precision('high')

if '__main__' == __name__:
    seed = 42
    set_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    model_config = GPT2Config()
    train_config = TrainConfig()

    dataset_dir = './data/edu_fineweb10B'

    B = train_config.B
    T = train_config.T

    trainloader = GPTDataLoader(B=B, T=T, dir_path=dataset_dir)
    valloader = GPTDataLoader(B=B, T=T, dir_path=dataset_dir, split='val')

    model = NdGPT2(config=model_config)
    model = torch.compile(model)
    model.eval()
    model.to(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.max_lr, fused=True)

    trained_model = train_gpt2(model=model,
                               trainloader=trainloader,
                               valloader=valloader,
                               optimizer=optimizer,
                               train_config=train_config)