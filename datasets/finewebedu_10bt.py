import torch
import os
import numpy as np
from typing import Tuple

class FineWebEdu10BT():
    def __init__(self, batch_size: int, seq_len: int, dir_path: str, split: str = 'train') -> None:
        self.B = batch_size
        self.T = seq_len

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
        self.position = 0

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