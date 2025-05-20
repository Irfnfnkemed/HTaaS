from contextlib import contextmanager
import math
from typing import Sized, Tuple
from torch.utils.data import DataLoader, Sampler, Dataset
import torch
from .env import *


class DataDistState:
    
    _epoch: int
    _index: int
    _len_dataset: int
    _drop_last: bool
    _total_epoch: int
    _world_size: int
    _stop_accum: bool
    _cur_bsz: int
    
    def __init__(self, total_epoch: int, len_dataset: int, drop_last=False):
        self._epoch = 0
        self._index = 0
        self._total_epoch = total_epoch
        self._drop_last = drop_last
        self._cur_bsz = 0
        self._world_size = world_size()
        self._stop_accum = False
        self._len_dataset = len_dataset
        
    @property
    def stop(self) -> bool:
        return self._index >= self._len_dataset or self._stop_accum
    
    def change_bs(self, cur_bsz: int):
        self._stop_accum = True
        self._cur_bsz = cur_bsz
    
    def start_accum(self):
        self._stop_accum = False
    
    def accumulate(self, batch_size: int):
        self._index += self._world_size * batch_size
        self._cur_bsz = batch_size
    
    def finish_and_update(self) -> bool:
        if self._index >= self._len_dataset or (self._drop_last and self._index + self._cur_bsz * self._world_size > self._len_dataset):
            self._index = 0
            self._epoch += 1
            return True
        else:
            return False
        
    def finish_all_epochs(self) -> bool:
        return self._epoch >= self._total_epoch
    
    def get_state(self) -> Tuple[int, int]:
        return self._epoch, self._index

    def set_state(self, state: Tuple[int, int]):
        self._epoch, self._index = state
        
    


class ElasticDistSampler(Sampler):
    
    _dataset: Dataset
    _shuffle: bool
    _world_size: int
    _rank: int
    _epoch: int
    _index: int    
    
    def __init__(self, dataset: Dataset, shuffle: bool = True):
        self._dataset = dataset
        self._shuffle = shuffle
        self._world_size = world_size()
        self._rank = rank()
        self._epoch = 0
        self._index = 0

    def __iter__(self):
        if self._shuffle:
            g = torch.Generator()
            g.manual_seed(hash((self._epoch, self._index // len(self._dataset)))) # type: ignore
            indices = torch.randperm(len(self._dataset), generator=g).tolist() # type: ignore
        else:
            indices = list(range(len(self._dataset))) # type: ignore

        # Subsample.
        base_index = self._index % len(self._dataset) # type: ignore
        local_indices = indices[base_index + self._rank :: self._world_size]

        # Add extra samples to make it evenly divisible.
        if len(local_indices) < len(self):
            local_indices.append(indices[self._rank])
        assert len(local_indices) == len(self)
        return iter(local_indices)

    def __len__(self):
        base_index = self._index % len(self._dataset) # type: ignore
        return math.ceil((len(self._dataset) - base_index) / self._world_size) # type: ignore

    def set_epoch(self, epoch: int, index: int = 0):
        self._epoch = epoch
        self._index = index
        

class ElasticDataLoader(DataLoader):
    
    _state: DataDistState
    _bs: int
    _update_bs: bool
    
    
    def __init__(self, dataset, epochs, **kwargs):
        if kwargs.get("batch_sampler") is not None or kwargs.get("sampler") is not None:
            raise ValueError("ElasticDataLoader does not support custom 'sampler' or 'batch_sampler'")
        if kwargs.get("batch_size") is None:
            kwargs["batch_size"] = 1
        shuffle = kwargs.get("shuffle", False)
        drop_last = kwargs.get("drop_last", False)
        kwargs["shuffle"] = False # shuffle in sampler, thus needn't shuffle in dataloader
        kwargs["sampler"] = ElasticDistSampler(dataset, shuffle=shuffle)
        self._state = DataDistState(epochs, len(dataset), drop_last=drop_last)
        self._bs = kwargs["batch_size"]
        self._update_bs = False
        super().__init__(dataset, **kwargs)
        
    def set_new_bs(self, new_batch_size: int):
        self._bs = new_batch_size
        self._update_bs = True
        self._state.change_bs(new_batch_size)

    def __iter__(self):
        while not self._state.finish_and_update():
            self.sampler.set_epoch(self._state.get_state()[0], self._state.get_state[1]) # type: ignore
            self.batch_sampler.batch_size = self._bs # type: ignore
            self._state.start_accum()
            for batch in super().__iter__():
                self._state.accumulate(self._bs)
                yield batch
                if self._state.stop:
                    break
    @contextmanager
    def gen_static_data(self):
        iteration = super().__iter__()
        index = self._state._index
        self.sampler.set_epoch(self._state.get_state[0], 0) # type: ignore
        try:
            yield iteration
        finally:
            self.sampler.set_epoch(self._state.get_state[0], index) # type: ignore
            
    @property
    def finish_all_epochs(self) -> bool:
        return self._state.finish_all_epochs()
    
    def get_state(self) -> Tuple[int, int]:
        return self._state.get_state()
    
    def load_ckpt(self, state: Tuple[int, int]):
        self._state.set_state(state)
                