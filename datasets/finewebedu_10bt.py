import torch
import os
import numpy as np
from typing import Tuple

class FineWebEdu10BT:
    def __init__(
        self, 
        batch_size: int, 
        seq_len: int, 
        dir_path: str, 
        split: str = 'train'
    ) -> None:
        
        self.batch_size = batch_size
        self.seq_len = seq_len

        assert split in {'train', 'val'}
        
        shards = os.listdir(dir_path)
        shards = [shard for shard in shards if split in shard]
        shards = sorted(shards)
        shards = [os.path.join(dir_path, shard) for shard in shards]
        self.shards = shards

        self.max_steps = self.steps_per_epoch()

        assert len(shards) > 0, "No shards found"

        self._reset()

    def _load_tokens(self, filename: str) -> torch.LongTensor:
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        return torch.from_numpy(npt).long()
    
    def _reset(self):
        self.current_shard = 0
        self.tokens = self._load_tokens(self.shards[self.current_shard])
        self.position = 0

    def get_batch(self) -> Tuple[torch.Tensor, torch.LongTensor]:
        batch_size, seq_len = self.batch_size, self.seq_len

        if self.position + (batch_size * seq_len + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_tokens(self.shards[self.current_shard])
            self.position = 0

            if len(self.tokens) < (batch_size * seq_len + 1):
                return self.get_batch()

        buffer = self.tokens[self.position : self.position + batch_size * seq_len + 1]
        x = buffer[:-1].view(batch_size, seq_len)
        y = buffer[1:].view(batch_size, seq_len)

        self.position += batch_size * seq_len
        return x, y

    def steps_per_epoch(self) -> int:
        max_steps = 0
        
        for shard_path in self.shards:
            tokens = np.load(shard_path)
            num_tokens = tokens.shape[0]
            max_steps += (num_tokens - 1) // (self.batch_size * self.seq_len)

        return max_steps
