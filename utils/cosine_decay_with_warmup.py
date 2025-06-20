import math

def cosine_decay_with_warmup(
          step: int, 
          warmup_steps: int, 
          max_steps: int, 
          max_lr: float, 
          min_lr: float
          ):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        
        if step > max_steps:
            return min_lr
        
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)

        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        return min_lr + coeff * (max_lr - min_lr)