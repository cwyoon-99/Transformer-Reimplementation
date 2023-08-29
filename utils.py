import re

from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler


def preprocess(dataset, src, tg):
    
    data_list = []
    for i in dataset:
        temp_dict = {}

        item = i["translation"]

        # For German, various white-space characters exist. rex replaces all the white-space characters to avoid some exceptions.
        src_text = re.sub(r'\s+', ' ', item[src])
        tg_text = re.sub(r'\s+', ' ', item[tg])

        temp_dict[src] = src_text.lower()
        temp_dict[tg] = tg_text.lower()

        data_list.append(temp_dict)

    return data_list

def calc_lr(step, dim_embed, warmup_steps):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

class TransformerScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups
