from contextlib import contextmanager
import math
from torch.utils.data import DataLoader, Sampler, Dataset
import torch
from .env import *

class ElasticDistSampler(Sampler):
    
    dataset: Dataset
    shuffle: bool
    world_size: int
    rank: int
    epoch: int
    index: int    
    
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.world_size = world_size()
        self.rank = rank()
        self.epoch = 0
        self.index = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(hash((self.epoch, self.index // len(self.dataset))))
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Subsample.
        base_index = self.index % len(self.dataset)
        local_indices = indices[base_index + self.rank :: self.world_size]

        # Add extra samples to make it evenly divisible.
        if len(local_indices) < len(self):
            local_indices.append(indices[self.rank])
        assert len(local_indices) == len(self)
        return iter(local_indices)

    def __len__(self):
        base_index = self.index % len(self.dataset)
        return math.ceil((len(self.dataset) - base_index) / self.world_size)

    def set_epoch(self, epoch, index=0):
        self.epoch = epoch
        self.index = index
        
class DataDistState:
    
    epoch: int
    index: int
    len_dataset: int
    drop_last: bool
    total_epoch: int
    world_size: int
    stop_accum: bool
    cur_bsz: int
    
    def __init__(self, total_epoch: int, len_dataset: int, drop_last=False):
        self.epoch = 0
        self.index = 0
        self.total_epoch = total_epoch
        self.drop_last = drop_last
        self.cur_bsz = 0
        self.world_size = world_size()
        self.stop_accum = False
        self.len_dataset = len_dataset
        
    @property
    def stop(self) -> bool:
        return self.index >= self.len_dataset or self.stop_accum
    
    def change_bs(self, cur_bsz: int):
        self.stop_accum = True
        self.cur_bsz = cur_bsz
    
    def start_accum(self):
        self.stop_accum = False
    
    def accumulate(self, batch_size: int):
        self.index += self.world_size * batch_size
        self.cur_bsz = batch_size
    
    def finish_and_update(self) -> bool:
        if self.index >= self.len_dataset or (self.drop_last and self.index + self.cur_bsz * self.world_size > self.len_dataset):
            self.index = 0
            self.epoch += 1
            return True
        else:
            return False
        
    

class ElasticDataLoader(DataLoader):
    
    state: DataDistState
    bs: int
    update_bs: bool
    
    
    def __init__(self, dataset, epochs, batch_size=1, shuffle=False, drop_last=False, **kwargs):
        if kwargs.get("batch_sampler") is not None or kwargs.get("sampler") is not None:
            raise ValueError("ElasticDataLoader does not support custom 'sampler' or 'batch_sampler'")
        kwargs["sampler"] = ElasticDistSampler(dataset, shuffle=shuffle)
        self.state = DataDistState(epochs, len(dataset), drop_last=drop_last)
        self.bs = batch_size
        self.update_bs = False
        super().__init__(dataset, batch_size, shuffle=False, drop_last=drop_last, **kwargs)
        
    def set_new_bs(self, new_batch_size: int):
        self.bs = new_batch_size
        self.update_bs = True
        self.state.change_bs(new_batch_size)

    def __iter__(self):
        while not self.state.finish_and_update():
            self.sampler.set_epoch(self.state.epoch, index=self.state.index)
            self.batch_sampler.batch_size = self.bs
            self.state.start_accum()
            for batch in super().__iter__():
                self.state.accumulate(self.bs)
                yield batch
                if self.state.stop:
                    break
    @contextmanager
    def gen_static_data(self):
        iteration = super().__iter__()
        index = self.state.index
        self.sampler.set_epoch(self.state.epoch, index=0)
        try:
            yield iteration
        finally:
            self.sampler.set_epoch(self.state.epoch, index=index)
                