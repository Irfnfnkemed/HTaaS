import math
from typing import List, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from . import env


class DataLoaderArguments:
    def __init__(self, dataset, num_workers=0, pin_memory=False, shuffle=False, drop_last=False, **kwargs):
        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_args(self) -> Dict[str, Any]:
        args = vars(self).copy()
        args['shuffle'] = False
        args.pop('batch_size', None)
        args.pop('sampler', None)
        return args

    def save_state(self) -> Dict[str, Any]:
        state = vars(self).copy()
        state.pop('dataset', None)
        return state

    def load_state(self, state: Dict[str, Any]):
        for key, value in state.items():
            setattr(self, key, value)


class GradMonitor:

    def __init__(self):
        self._model: DistributedDataParallel = None

        self.mean_ept = torch.tensor(0.0)
        self.variance_ept = torch.tensor(0.0)
        self.mean_epb = torch.tensor(0.0)
        self.variance_epb = torch.tensor(0.0)
        self.ept_average = torch.tensor(0.0)
        self.epb_average = None

        # Args for measuring epb when world_size=1
        self.epb_list = []
        self.accumulation_steps = 1
        self.now_steps = 0

        self.gamma = 0.9
        self.grad_max_dim = 50000000
        self.epb_chunks = 100
        self.ept_ema = 0.8
        self.epb_ema = 0.95

        self.grad_dim = 0
        self.grad_mask = []
        self.grad_mask_len = []
        
        self.para_grad = None
        
        self.stop_ept = False
        self.stop_epb = False

    def associate(self, model: DistributedDataParallel):
        self._model = model
        self.grad_dim = 0
        for param in self._model.parameters():
            self.grad_dim += param.data.numel()
        if self.grad_dim > self.grad_max_dim:
            torch.manual_seed(42)
            rate = self.grad_max_dim / self.grad_dim
            self.grad_dim = 0
            for param in self._model.parameters():
                sample = math.ceil(param.numel() * rate)
                self.grad_dim += sample
                sample_indices = torch.randint(0, param.numel(), (sample,)).to(dist.get_rank())
                self.grad_mask.append(torch.tensor(sample_indices))
                self.grad_mask_len.append(sample)
        self.para_grad = torch.zeros(self.grad_dim).to(dist.get_rank())
        self.mean_epb = torch.zeros(self.grad_dim).to(dist.get_rank())
        self.variance_epb = torch.zeros(self.grad_dim).to(dist.get_rank())

    def set_accumulation_steps(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.now_steps = 0

    def _monitor_grad(self):
        self.para_grad.zero_()
        cur_index = 0
        for index, param in enumerate(self._model.parameters()):
            if len(self.grad_mask) > 0:
                if param.grad is not None:
                    self.para_grad[cur_index: cur_index + self.grad_mask_len[index]] = param.grad.view(-1)[self.grad_mask[index]]
                cur_index += self.grad_mask_len[index]
            else:
                num_ele = param.data.numel()
                if param.grad is not None:
                    self.para_grad[cur_index: cur_index + num_ele] = param.grad.view(-1)
                cur_index += num_ele

    def monitor_ept(self):
        if self.stop_ept:
            return
        self._monitor_grad()
        self.mean_ept = self.gamma * self.mean_ept + (1 - self.gamma) * self.para_grad
        self.variance_ept = self.gamma * self.variance_ept + (1 - self.gamma) * (self.para_grad ** 2)
        ept = self._ept()
        self.ept_average = self.ept_average * self.ept_ema + ept * (1 - self.ept_ema)
        
    def monitor_epb(self, flag: bool):
        if self.stop_epb:
            return
        if dist.get_world_size() > 1 and flag:
            self._monitor_grad()
            from_rank = dist.get_rank() - 1
            to_rank = dist.get_rank() + 1
            self.mean_epb.zero_()
            self.variance_epb.zero_()

            if from_rank >= 0:
                result = dist.recv(self.mean_epb, src=from_rank, tag=1)
                assert result == from_rank
                result = dist.recv(self.variance_epb, src=from_rank, tag=2)
                assert result == from_rank

            if dist.get_rank() == 0:
                self.mean_epb = self.para_grad.clone()
                self.variance_epb = self.para_grad ** 2
            else:
                self.mean_epb = self.mean_epb + self.para_grad
                self.variance_epb = self.variance_epb + self.para_grad ** 2

            if to_rank < dist.get_world_size():
                dist.send(self.mean_epb, dst=to_rank, tag=1)
                dist.send(self.variance_epb, dst=to_rank, tag=2)

            dist.barrier()
            epb = self._epb().clone().detach().to(dist.get_rank())
            epb_gather_list = [torch.zeros(1, dtype=torch.float32, device=dist.get_rank()) for _ in range(dist.get_world_size())]
            dist.all_gather(epb_gather_list, epb)
            epb_gather_list = [tensor.to('cpu').item() for tensor in epb_gather_list]
            epb_corr = self._correct_epb(epb_gather_list)
            if epb_corr is not None:
                if self.epb_average is None:
                    self.epb_average = epb_corr
                elif 1 < epb_corr < self.epb_average * 10:
                    self.epb_average = self.epb_average * self.epb_ema + epb_corr * (1 - self.epb_ema)
        elif dist.get_world_size() == 1 and not flag:
            self._monitor_grad()
            self.now_steps += 1
            self.variance_epb = self.variance_epb + (self.para_grad - self.mean_epb) ** 2
            self.mean_epb = self.para_grad.clone()
            self.epb_list.append(self._epb())
            if self.now_steps == self.accumulation_steps:
                self.now_steps = 0
                self.mean_epb.zero_()
                self.variance_epb.zero_()
                epb_corr = self._correct_epb(self.epb_list)
                if epb_corr is not None:
                    if self.epb_average is None:
                        self.epb_average = epb_corr
                    elif 1 < epb_corr < self.epb_average * 10:
                        self.epb_average = self.epb_average * self.epb_ema + epb_corr * (1 - self.epb_ema)
                self.epb_list.clear()

    def _correct_epb(self, epb_list: List[float]):
        for i, v in enumerate(epb_list):
            if v > i + 1.1:
                return None  # Invalid epb-measurement
        a = np.arange(1, len(epb_list) + 1)
        b = np.array(epb_list)
        x = 1 - b / a
        y = b * (1 - 1 / a)
        epb_corr = ((x * y).mean() - x.mean() * y.mean()) / ((x ** 2).mean() - (x.mean()) ** 2)
        if epb_corr < 1:
            return None  # Invalid epb-measurement
        return torch.tensor(epb_corr)

    def _ept(self) -> torch.Tensor:
        ratio = self.mean_ept ** 2 / (self.variance_ept + 1e-30)
        ratio = ratio[(ratio < 1) & (ratio > 0)]
        ept = torch.log10(1 / torch.mean(ratio))
        if torch.isnan(ept) or torch.isinf(ept):
            ept = torch.tensor(0.0)
        return ept

    def _epb(self) -> torch.Tensor:
        part_size = min(self.grad_max_dim, self.grad_dim) // self.epb_chunks
        sub_mean = torch.split(self.mean_epb, part_size)
        sub_mean = torch.tensor([(chunk ** 2).sum() for chunk in sub_mean])
        sub_variance = torch.split(self.variance_epb, part_size)
        sub_variance = torch.tensor([chunk.sum() for chunk in sub_variance])
        ratio = sub_mean / (sub_variance + 1e-30)
        ratio = ratio[ratio > 0]
        if dist.get_world_size() > 1:
            epb = (dist.get_rank() + 1) / torch.mean(ratio)
        else:
            epb = self.now_steps / torch.mean(ratio)
        return epb

    @property
    def ept(self) -> torch.Tensor:
        return self.ept_average

    @property
    def epb(self):
        if self.epb_average is None:
            return None
        return torch.log2(self.epb_average - 1)


class Adjuster:

    def __init__(self):
        self._optimizer: torch.optim.Optimizer = None
        self._grad_monitor: GradMonitor = None

        # config of ept
        self.warmup_ept = 20
        self.adjust_interval = 5
        self.low_error = -0.03
        self.high_error = 0.05
        self.eta = 3
        self.standard_ept = 1.3
        self.now_steps = 0
        self.fit_lr_steps = 0
        self.freeze_lr_bound = 15
        self.freeze_lr = False

        # config of epb
        self.max_step = 30
        self.lower_bound = -0.5
        self.upper_bound = 0.5
        self.max_bs = 1000
        self.min_bs = 10
        self.max_global_bs = 0
        self.bias = 0.0
        self.accumulation_step = 1
        self.freeze_bs = False

    def associate(self, optimizer: torch.optim.Optimizer, grad_monitor: GradMonitor):
        self._optimizer = optimizer
        self._grad_monitor = grad_monitor

    def adjust_lr(self):
        self.now_steps += 1
        if self.freeze_lr:
            return
        if self.now_steps > self.warmup_ept and self.now_steps % self.adjust_interval == 0:
            ept = self._grad_monitor.ept
            rate = (10 ** ((self.standard_ept - ept) / self.eta)).item()
            print(ept)
            if self.low_error <= ept - self.standard_ept <= self.high_error or \
                    math.isnan(rate) or math.isinf(rate):
                rate = 1.0
                self.fit_lr_steps += 1
            else:
                self.fit_lr_steps = max(0, self.fit_lr_steps - 1)
            for param_group in self._optimizer.param_groups:
                print(param_group['lr'])
                param_group['lr'] *= rate
            if self.fit_lr_steps > self.freeze_lr_bound:
                self.freeze_lr = True
                self._grad_monitor.stop_ept = True
                

    def adjust_bs(self, bs: int) -> int:
        epb = self._grad_monitor.epb
        if self.freeze_bs:
            return bs
        if self.lower_bound + self.bias <= epb <= self.upper_bound + self.bias:
            new_bs = bs
        elif epb < self.lower_bound + self.bias:
            new_bs = int(2 ** (epb - self.lower_bound - self.bias) * bs)
            new_bs = min(new_bs / 0.7, bs)
        else:
            new_bs = int(2 ** (epb - self.upper_bound - self.bias) * bs)
            new_bs = max(new_bs * 0.7, bs)
        if new_bs > self.max_global_bs:
            self.freeze_bs = True
            self._grad_monitor.stop_epb = True
        return min(new_bs, self.max_global_bs)
    
    
    def set_accumulate_steps(self, new_accumulate_step):
        self.accumulation_step = new_accumulate_step

    def adjust_accumulate_step(self, new_accumulate_step):
        if self.accumulation_step != new_accumulate_step:
            rate = self.accumulation_step / new_accumulate_step
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= rate
            self.accumulation_step = new_accumulate_step

    def set_bs_config(self, bias: float):
        self.bias = bias + math.log2(self.get_num())

    def get_num(self):
        return self.accumulation_step if env.world_size() == 1 else env.world_size()
    
    def set_init_bs_config(self, max_bs: int, max_global_bs: int):
        self.max_bs = max_bs
        self.max_global_bs = max_global_bs
