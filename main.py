from models.gpt2 import GPT2
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torchdata.stateful_dataloader import StatefulDataLoader
from config import GPT2Config, TrainConfig
import torch
import torch.nn as nn
from experiment.train import train_gpt2
from utils.parameter_count import parameter_count
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets.distributed import split_dataset_by_node
import os

torch.set_float32_matmul_precision('high')

if '__main__' == __name__:
    model_config = GPT2Config()
    train_config = TrainConfig()

    ckpt_path = f'./checkpoints/models/{train_config.model_name}/{train_config.model_name}.pt'

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})

    dataset = load_dataset(train_config.dataset_name, split='train', streaming=True)
    dataset = dataset.map(
        lambda samples: tokenizer(
            samples["text"], 
            truncation=True, 
            padding='max_length', 
            max_length=train_config.seq_len), 
            remove_columns=dataset.features
            )
    dataset = dataset.with_format('torch')
    

    if train_config.use_ddp:
        # torchrun --standalone --nproc_per_node=gpu main.py
        init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'

        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

        if rank == 0:
            print(f'Using DDP with NCCL (CUDA)')
    else: 
        local_rank = 0
        rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using device: {device}')


    model = GPT2(config=model_config)
    model.to(device)
    if train_config.use_ddp:
        model = DDP(model, device_ids=[local_rank])

    model.compile(dynamic=True)

    dataloader = StatefulDataLoader(
        dataset=dataset, 
        batch_size=train_config.batch_size, 
        num_workers=train_config.num_workers, 
        pin_memory=True
        )

    if rank == 0:
        num_params = parameter_count(model=model)
        print(f"Number of parameters for this model is {num_params}")

    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            nodecay_params.append(param)
        else: decay_params.append(param)

    model_params = [
        {'params': decay_params, 'weight_decay': train_config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(model_params, betas=(0.9, 0.95), fused=True)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    if os.path.exists(ckpt_path):
        print(f'Loading checkpoint: {ckpt_path}')
        state_dict = torch.load(ckpt_path)
        train_config.start_step = state_dict['step']
        train_config.optimization_step = state_dict['optimization_step']
        model.load_state_dict(state_dict['model'], strict=True)
        optimizer.load_state_dict(state_dict['optimizer'])
        dataloader.load_state_dict(state_dict['dataloader'])

    trained_model = train_gpt2(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        tokenizer=tokenizer,
        train_config=train_config,
        rank=rank,
        world_size=world_size
        )
    
    if train_config.use_ddp:
        destroy_process_group()