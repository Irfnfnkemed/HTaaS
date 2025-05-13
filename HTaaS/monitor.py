from contextlib import contextmanager
from typing import Dict, List
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import *
from .data import DynamicDataLoader

class GradMonitor:
    
    model: DDP
    dataloader: DynamicDataLoader
    
    # monitor the inital gradient
    grad_dim: int
    grad_buffer: torch.Tensor
    grad_mask: List[torch.Tensor]
    
    # monitor GRt
    momentum: torch.Tensor
    variance: torch.Tensor
    GRt = float
    
    # monitor GRs
    accum_m: torch.Tensor
    accum_v: torch.Tensor
    acc_GRs_list: List[float]
    GRs = float
    
    

    def __init__(self, model: DDP, dataloader: DynamicDataLoader):
        """
        Initialize the GradMonitor class.

        Args:
            model: The DDP model whose gradient will be monitored.
            dataloader: The dataloader used to train.
        """
        self.model = model
        self.dataloader = dataloader
        self.grad_dim = 0
        for param in self.model.parameters():
            self.grad_dim += param.data.numel()
        # generate the grad mask
        self.grad_mask = []
        if self.grad_dim > MAX_GRAD_DIM:
            rate = MAX_GRAD_DIM / self.grad_dim
            self.grad_dim = 0
            for param in self.model.parameters():
                numel = param.data.numel()
                sample_size = torch.ceil(numel * rate)
                self.grad_dim += sample_size
                sample_indices = torch.randint(0, numel, (sample_size,)).to(param.device)
                self.grad_mask.append(torch.tensor(sample_indices))
        # prepare the space
        self.grad_buffer = torch.zeros(self.grad_dim).to(param.device)
        self.momentum = torch.zeros(self.grad_dim).to(param.device)
        self.variance = torch.zeros(self.grad_dim).to(param.device)
        self.GRt = 0
        part_size = self.grad_dim // GRS_CHUNKS
        total_size = GRS_CHUNKS * part_size
        self.accum_m = torch.zeros(total_size).to(param.device)
        self.accum_v = torch.zeros(total_size).to(param.device) 
        self.GRs = 0
        self.acc_GRs_list = []

    def _monitor_grad(self):
        """
        Monitors and updates the gradient buffer based on the gradients of the model parameters.
        """
        self.grad_buffer.zero_()
        cur_index = 0
        if self.grad_dim <= MAX_GRAD_DIM:
            for _, param in enumerate(self.model.parameters()):
                num_ele = param.data.numel()
                if param.grad is not None:
                    self.grad_buffer[cur_index: cur_index + num_ele] = param.grad.view(-1)
                cur_index += num_ele
        else:
            for index, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    self.grad_buffer[cur_index: cur_index + len(self.grad_mask[index])] = param.grad.view(-1)[self.grad_mask[index]]
                cur_index += len(self.grad_mask[index])


    def monitor_GRt(self):
        # update m and v with EMA
        self._monitor_grad()
        self.momentum.mul_(BETA).addcmul_(1 - BETA, self.grad_buffer)
        self.variance.mul_(BETA).addcmul_(1 - BETA, self.grad_buffer.pow(2))
        # calculate GRt
        ratio = self.momentum.pow(2) / (self.variance + 1e-10)
        valid_mask = (ratio < 1) & (ratio > 0)
        sum_valid = torch.sum(ratio * valid_mask)
        count_valid = torch.sum(valid_mask) 
        GRt = torch.log10(count_valid / sum_valid)
        # update GRt with EMA
        self.GRt = GRT_EMA * self.GRt + (1 - GRT_EMA) * GRt.item()
        
    def monitor_GRs(self, need_sample_grad: bool):
        # monitor the gradient and update
        size = self.accum_m.shape[0]
        part_size = size // GRS_CHUNKS
        if need_sample_grad:
            self._monitor_grad()
        self.accum_m.add_(self.grad_buffer[:size])
        self.accum_v.add_(self.grad_buffer[:size].pow(2))
        # calculate GRs with current value
        accum_m_reshaped = self.accum_m.view(GRS_CHUNKS, part_size)
        accum_v_reshaped = self.accum_v.view(GRS_CHUNKS, part_size)
        ratio = accum_m_reshaped.pow(2).sum(dim=1) / (accum_v_reshaped.sum(dim=1) + 1e-10)
        valid_mask = ratio > 0
        sum_valid = torch.sum(ratio * valid_mask)
        count_valid = torch.sum(valid_mask) 
        self.acc_GRs_list.append((1 + len(self.acc_GRs_list)) * count_valid / sum_valid) 
    
    @contextmanager
    def env_GRs(self):
        self.acc_GRs_list.clear()
        self.accum_m.zero_()
        self.accum_v.zero_()
        try:
            yield
        finally:
            # correct GRs with math method
            x = []
            y = []
            for i, v in enumerate(self.acc_GRs_list):
                if v > i + 1.1:
                    continue  # invalid GRs measurement
                x.append(1 - v / (i + 1))
                y.append(v - 1)
            if len(x) < 2:
                print("Warning: invalid measurement in GRs, ignore.")
            x = np.array(x)
            y = np.array(y)
            GRs_corr = np.log2(((x * y).mean() - x.mean() * y.mean()) / ((x ** 2).mean() - (x.mean()) ** 2))
            # this is used in debug: the fitting value of intercept should be close to 0
            # intercept =  y.mean() - x.mean() * (2 ** GRs_corr)
            self.GRs = GRs_corr
            
    @property
    def GRt(self) -> float:
        return self.GRt

    @property
    def GRs(self) -> float:
        return self.GRs

