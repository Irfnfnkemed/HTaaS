from dataclasses import dataclass
import math
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .data import ElasticDataLoader
from .comm import *
from .env import *
from .config import *


from contextlib import contextmanager
from typing import Any, Dict, List
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import *

class GradMonitor:
    
    _model: DDP
    
    # monitor the inital gradient
    _grad_dim: int
    _grad_buffer: torch.Tensor
    _grad_mask: List[torch.Tensor]
    
    # monitor GRt
    _momentum: torch.Tensor
    _variance: torch.Tensor
    _GRt: float
    
    # monitor GRs
    _accum_m: torch.Tensor
    _accum_v: torch.Tensor
    _acc_GRs_list: List[float]
    _GRs: float
    
    

    def __init__(self, model: DDP):
        """
        Initialize the GradMonitor class.

        Args:
            model: The DDP model whose gradient will be monitored.
            dataloader: The dataloader used to train.
        """
        self._model = model
        self._grad_dim = 0
        for param in self._model.parameters():
            self._grad_dim += param.data.numel()
        # generate the grad mask
        self._grad_mask = []
        if self._grad_dim > MAX_GRAD_DIM:
            rate = MAX_GRAD_DIM / self._grad_dim
            self._grad_dim = 0
            for param in self._model.parameters():
                numel = param.data.numel()
                sample_size = math.ceil(numel * rate)
                self._grad_dim += sample_size
                sample_indices = torch.randint(0, numel, (sample_size,)).to(param.device)
                self._grad_mask.append(torch.tensor(sample_indices))
        # prepare the space
        self._grad_buffer = torch.zeros(self._grad_dim).to(param.device)
        self._momentum = torch.zeros(self._grad_dim).to(param.device)
        self._variance = torch.zeros(self._grad_dim).to(param.device)
        self._GRt = 0.0
        part_size = self._grad_dim // GRS_CHUNKS
        total_size = GRS_CHUNKS * part_size
        self._accum_m = torch.zeros(total_size).to(param.device)
        self._accum_v = torch.zeros(total_size).to(param.device) 
        self._GRs = 0.0
        self._acc_GRs_list = []
    
    def save_ckpt(self) -> Dict[str, Any]:
        return {
            'momentum': self._momentum,
            'variance': self._variance,
            'GRt': self._GRt,
            'GRs': self._GRs,
        }
    
    def load_ckpt(self, ckpt: Dict[str, Any]):
        self._momentum = ckpt['momentum']
        self._variance = ckpt['variance']
        self._GRt: float = ckpt['GRt']
        self._GRs: float = ckpt['GRs']

    def _monitor_grad(self):
        """
        Monitors and updates the gradient buffer based on the gradients of the model parameters.
        """
        self._grad_buffer.zero_()
        cur_index = 0
        if self._grad_dim <= MAX_GRAD_DIM:
            for _, param in enumerate(self._model.parameters()):
                num_ele = param.data.numel()
                if param.grad is not None:
                    self._grad_buffer[cur_index: cur_index + num_ele] = param.grad.view(-1)
                cur_index += num_ele
        else:
            for index, param in enumerate(self._model.parameters()):
                if param.grad is not None:
                    self._grad_buffer[cur_index: cur_index + len(self._grad_mask[index])] = param.grad.view(-1)[self._grad_mask[index]]
                cur_index += len(self._grad_mask[index])


    def monitor_GRt(self):
        # update m and v with EMA
        self._monitor_grad()
        self._momentum.mul_(BETA / (1 - BETA)).add_(self._grad_buffer).mul_(1 - BETA)
        self._variance.mul_(BETA / (1 - BETA)).add_(self._grad_buffer.pow(2)).mul_(1 - BETA)
        # calculate GRt
        ratio = self._momentum.pow(2) / (self._variance + 1e-10)
        valid_mask = (ratio < 1) & (ratio > 0)
        sum_valid = torch.sum(ratio * valid_mask)
        count_valid = torch.sum(valid_mask) 
        GRt = torch.log10(count_valid / sum_valid)
        # update GRt with EMA
        self._GRt = GRT_EMA * self._GRt + (1 - GRT_EMA) * GRt.item()
        
    def monitor_GRs(self, need_sample_grad: bool):
        # monitor the gradient and update
        size = self._accum_m.shape[0]
        part_size = size // GRS_CHUNKS
        if need_sample_grad:
            self._monitor_grad()
        self._accum_m.add_(self._grad_buffer[:size])
        self._accum_v.add_(self._grad_buffer[:size].pow(2))
        # calculate GRs with current value
        accum_m_reshaped = self._accum_m.view(GRS_CHUNKS, part_size)
        accum_v_reshaped = self._accum_v.view(GRS_CHUNKS, part_size)
        ratio = accum_m_reshaped.pow(2).sum(dim=1) / (accum_v_reshaped.sum(dim=1) + 1e-10)
        valid_mask = ratio > 0
        sum_valid = torch.sum(ratio * valid_mask)
        count_valid = torch.sum(valid_mask) 
        self._acc_GRs_list.append(((1 + len(self._acc_GRs_list)) * count_valid / sum_valid).item()) 
    
    @contextmanager
    def env_GRs(self):
        self._acc_GRs_list.clear()
        self._accum_m.zero_()
        self._accum_v.zero_()
        try:
            yield
        finally:
            # correct GRs with math method
            x = []
            y = []
            for i, v in enumerate(self._acc_GRs_list):
                if v > i + 1.1:
                    continue  # invalid GRs measurement
                x.append(1 - v / (i + 1))
                y.append(v - 1)
            if len(x) < 2:
                print("Warning: invalid measurement in GRs, ignore.")
            x = np.array(x)
            y = np.array(y)
            slope, intercept = np.polyfit(x, y, 1)
            GRs_corr = np.log2(slope)
            self._GRs = GRs_corr
            
    @property
    def GRt(self) -> float:
        return self._GRt

    @property
    def GRs(self) -> float:
        return self._GRs


class Adjuster:
    
    _optimizer: optim.Optimizer
    _max_local_bs: int
    _max_global_bs: float
    _GRs_target: float
    _GRt_target: float
    _ideal_global_bs: float
    
    def __init__(self, optimizer: optim.Optimizer):
        self._optimizer = optimizer
        self._max_local_bs = 0
        self._max_global_bs = 0.0
        self._GRt_target = GRT_TARGET
        self._GRs_target = GRS_TARGET
        self._ideal_global_bs = 0.0
    
    def save_ckpt(self) -> Dict[str, Any]:
        return {
            'max_local_bs': self._max_local_bs,
            'max_global_bs': self._max_global_bs,
            'ideal_global_bs': self._ideal_global_bs,
        }
        
    def load_ckpt(self, ckpt: Dict[str, Any]):
        self._max_local_bs = ckpt['max_local_bs']
        self._max_global_bs = ckpt['max_global_bs']
        self._ideal_global_bs = ckpt['ideal_global_bs']
        
        
    def update_GRs_target(self, GRs_target: float):
        self._GRs_target = GRs_target
        
    def set_init_config(self, max_local_bs: int, ideal_global_bs: float):
        self._max_local_bs = max_local_bs
        self._max_global_bs = ideal_global_bs * MAX_BS_RATIO
        self._ideal_global_bs = ideal_global_bs
        
    def adjust_lr(self):
        pass
                
    def adjust_ideal_bs(self, GRs: float) -> float:
        self._ideal_global_bs = min(self._max_global_bs, self._ideal_global_bs * pow(2, GRs - self._GRs_target))
        return self._ideal_global_bs
    
    def adjust_local_bs(self) -> Tuple[int, int]:
        n = world_size()
        accum_step = math.ceil(self._ideal_global_bs / self.max_local_bs / n)
        local_bs = min(self._max_local_bs, int(self._ideal_global_bs / n / accum_step))
        self._ideal_global_bs = float(accum_step * n * local_bs) 
        return accum_step, local_bs
    
    @property
    def GRs_target(self):
        return self._GRs_target
    
    @property
    def max_local_bs(self):
        return self._max_local_bs

    @property
    def ideal_devices(self) -> float:
        return self._ideal_global_bs / self._max_local_bs
    

@dataclass
class Counter:
    _count: int = 0
    _accum: int = 1
    
    def step(self):
        self._count += 1
        
    def reset(self):
        self._count = 0
    
    def set_accum(self, accum: int):
        """
        Set the accumulation step to the given value. This operation is only allowed when the counter's count is a multiple of the current accumulation step.

        Args:
            accum: The new accumulation step.
        """
        assert self.need_update
        self._count = self._count // self._accum * accum
        self._accum = accum
        
    @property
    def need_update(self) -> bool:
        """
        Whether the training needs to update parameters now.
        """
        return self._count % self._accum == 0
    
    @property
    def iter_count(self) -> int:
        """
        Calculate the number of completed accumulation steps.
        """
        return self._count // self._accum
    
    @property
    def accum(self) -> int:
        """
        Get the current accumulation step.
        """
        return self._accum
    
    


class Trainer(ABC):
    
    _model: DDP
    _trainloader: ElasticDataLoader
    _testloader: DataLoader
    _optimizer: optim.Optimizer
    _grad_monitor: GradMonitor
    _adjuster: Adjuster
    _counter: Counter
    _conn_to_processor: ClientInstance
    _job_id: int
    _device: torch.device
    _gpu_mem_utilization: float
    

    def __init__(self, model: torch.nn.Module, trainset: Dataset, 
                 trainloader_args: Dict[str, Any], testloader: DataLoader,
                 optimizer: Optional[torch.optim.Optimizer], total_epochs: int, 
                 gpu_mem_utilization: float = 0.85):
        if not dist.is_available() or not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=world_size(), rank=rank())
        # initialize the trainer
        self._model = DDP(model, device_ids=[local_rank()], output_device=local_rank())
        self._device = next(self._model.parameters()).device
        self._trainloader = ElasticDataLoader(trainset, total_epochs, **trainloader_args)
        self._testloader = testloader
        if optimizer is None:
            self._optimizer = optim.AdamW(self._model.parameters()) # default AdamW optimizer
        else:
            self._optimizer = optimizer
        self._grad_monitor = GradMonitor(self._model)
        self._adjuster = Adjuster(self._optimizer)
        self._counter = Counter()
        self._conn_to_processor = ClientInstance()
        self._gpu_mem_utilization = gpu_mem_utilization

        # connect to processor and get job-id
        tmp_buffer = torch.tensor(0).to(self._device)
        if rank() == 0:
            #print(processor_host(), processor_port())
            self._conn_to_processor.connect(processor_host(), processor_port())
            self._conn_to_processor.send('init', '')
            cmd, job_id = self._conn_to_processor.recv()
            assert cmd == 'init'
            tmp_buffer = torch.tensor(int(job_id)).to(self._device)
        dist.broadcast(tmp_buffer, src=0)
        dist.barrier()
        self._job_id = int(tmp_buffer.item())
        # load from ckpt or profile
        self.load_ckpt_or_profile()

    def save_checkpoint(self):
        checkpoint = {
            'model': self._model.module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'trainloader': self._trainloader.get_state(),
            'monitor': self._grad_monitor.save_ckpt(),
            'adjuster': self._adjuster.save_ckpt(),
        }
        torch.save(checkpoint, f'{WORK_DIR}/job_{self._job_id}/ckpt.pth')

    def load_ckpt_or_profile(self):
        if os.path.exists(f'{WORK_DIR}/job_{self._job_id}/ckpt.pth'):
            print("Loading from checkpoint")
            ckpt = torch.load(f'{WORK_DIR}/job_{self._job_id}/ckpt.pth', weights_only=True)
            self._model.module.load_state_dict(ckpt['model'])
            self._optimizer.load_state_dict(ckpt['optimizer'])
            self._trainloader.load_ckpt(ckpt['trainloader'])
            self._grad_monitor.load_ckpt(ckpt['monitor'])
            self._adjuster.load_ckpt(ckpt['adjuster'])
            accum_step, local_bs = self._adjuster.adjust_local_bs()
            self._counter.set_accum(accum_step)
            self._trainloader.set_new_bs(local_bs)
        else:
            self.profile()

    def profile(self):
        max_mem = torch.cuda.get_device_properties(self._model.device).total_memory * self._gpu_mem_utilization
        assert world_size() == 1
        self._model.train()
        GRs_list = []
        mem_list = []
        bs_list = []
        max_r2 = 0
        best_slope_GRs, best_intercept_GRs = 0.0, 0.0 
        slope_mem, intercept_mem = 0.0, 0.0
        try:
            for log_bs in range(15):
                # profile the memory-use and GRs 
                bs = 2 ** log_bs # local batch size
                torch.cuda.reset_peak_memory_stats()
                self._trainloader.set_new_bs(bs)
                with self._trainloader.gen_static_data() as datas_grs:
                    tmp_counter = Counter()
                    for _, data_grs in datas_grs:
                        tmp_counter.step()
                        input_grs = self.get_input(self._device, data_grs)
                        output_grs = self._model(input_grs)
                        loss_grs = self.get_loss(self._device, data_grs, output_grs) / tmp_counter.accum
                        loss_grs.backward()
                        if tmp_counter.need_update:
                            self._grad_monitor.monitor_GRs(True)
                            self._optimizer.zero_grad()
                            if tmp_counter.iter_count == GRS_ACC_TIMES:
                                break # end monitoring GRs
                GRs_list.append(self._grad_monitor.GRs)
                mem_list.append(torch.cuda.max_memory_allocated())
                bs_list.append(bs)                   
                # try to fit GRs = log(mu/B)
                beg_index = 0 if log_bs < PROFILE_WINDOW else log_bs - PROFILE_WINDOW + 1
                # bs_list is the local bs, need to mul with world size when fit GRs
                x = np.log2(np.array(bs_list[beg_index:]) * world_size()) 
                y = np.array(GRs_list[beg_index:])
                slope, intercept = np.polyfit(x, y, 1)
                y_fit = slope * x + intercept
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                if r2 > max_r2:
                    max_r2 = r2
                    best_slope_GRs, best_intercept_GRs = slope, intercept
                if r2 > 0.995:
                    break # end profiling
                # fit memory with bs, try to avoid OOM
                x = np.array(mem_list)
                y = np.array(bs_list)
                slope_mem, intercept_mem = np.polyfit(x, y, 1)
                y_fit = slope_mem * x + intercept_mem
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                print(r2)
                max_log_bs = np.log2((max_mem - intercept_mem) / slope_mem)
                if log_bs + 1 > max_log_bs:
                    break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pass # handle OOM
            else:
                raise
        
        self.update_GRs_target()
        max_log_bs = np.log2((max_mem - intercept_mem) / slope_mem)
        max_local_bs = pow(2, np.floor(max_log_bs))
        ideal_global_bs = pow(2, (self._adjuster.GRs_target - best_intercept_GRs) / best_slope_GRs)
        self._adjuster.set_init_config(max_local_bs, ideal_global_bs)
        self.adjust_resources()
        
        
    def run(self):
        epoch, index = self._trainloader.get_state()
        if epoch == 0 and index == 0:
            self.on_train_start()
        while not self._trainloader.finish_all_epochs:
            # print(f'[RANK{rank()}]: bs:{self._ideal_global_bs}, accumu: {self._accumulation_steps}')
            # print(f"[RANK{rank()}]: epoch-beg: {time.time()}")
            
            # training
            epoch, index = self._trainloader.get_state()
            if index == 0: 
                self.on_epoch_start(epoch)
            self._model.train()
            for data in self._trainloader:
                epoch, index = self._trainloader.get_state()
                self.on_batch_start(index)
                self._counter.step()
                input = self.get_input(self._device, data)
                self.on_forward_start(index, data, input)
                output = self._model(input)
                self.on_forward_end(index, data, input, output)
                loss = self.get_loss(self._device, data, output) / self._counter.accum
                self.on_backward_start(index, data, input, output, loss)
                loss.backward()
                self.on_backward_end(index, data, input, output, loss)
                if self._counter.need_update:
                    # monitor GRt
                    self._grad_monitor.monitor_GRt()
                    # adjust lr according to GRt
                    self._adjuster.adjust_lr()
                    self.on_step_start(index, data, input, output, loss)
                    self._optimizer.step()
                    self.on_step_end(index, data, input, output, loss)
                    self._optimizer.zero_grad()
                    if self._counter.iter_count % GRS_MONITOR_INTERVAL == 0:
                        with self._grad_monitor.env_GRs():
                            # leverage the monitored grad
                            tmp_counter.step()
                            self._grad_monitor.monitor_GRs(False)
                            self._optimizer.zero_grad()
                            # monitor GRs without para update
                            tmp_counter = Counter()
                            tmp_counter.set_accum(self._counter.accum)
                            with self._trainloader.gen_static_data() as datas_grs:
                                for _, data_grs in datas_grs:
                                    tmp_counter.step()
                                    input_grs = self.get_input(self._device, data_grs)
                                    output_grs = self._model(input_grs)
                                    loss_grs = self.get_loss(self._device, data_grs, output_grs) / tmp_counter.accum
                                    loss_grs.backward()
                                    if tmp_counter.need_update:
                                        self._grad_monitor.monitor_GRs(True)
                                        self._optimizer.zero_grad()
                                        if tmp_counter.iter_count == GRS_ACC_TIMES:
                                            break # end monitoring GRs
                self.on_batch_end(index, data, input, output, loss)
                if self._counter.iter_count % GRS_MONITOR_INTERVAL == 0:
                    self.update_GRs_target()
                    self._adjuster.adjust_ideal_bs(self._grad_monitor.GRs) # adjust ideal_bs
                    self.adjust_resources() # adjust resources according to GRs
                
            print(f"[RANK{rank()}]: epoch-train-finish: {time.time()}")

            # evaluating
            epoch, index = self._trainloader.get_state() # expect to be: (next epoch, 0)
            self._model.eval()
            self.evaluate(self._device, epoch - 1, self._model.module, self._testloader)

            print(f"[RANK{rank()}]: epoch-eval-finish: {time.time()}")
            self.on_epoch_end(epoch - 1)


        self.on_train_end()
        if rank() == 0:
            self._conn_to_processor.send('end', '')
            self._conn_to_processor.close()
        dist.barrier()
        self.exit()

    def update_GRs_target(self):
        # get current GRs_target maintained by engine
        GRs_target = torch.tensor(GRS_TARGET).to(self._device)
        if rank() == 0:
            self._conn_to_processor.send('GRs_target', '')
            cmd, standard = self._conn_to_processor.recv()
            assert cmd == 'GRs_target'
            GRs_target = torch.tensor(float(standard)).to(self._device)
        dist.broadcast(GRs_target, src=0)
        # update GRs_target in adjuster
        self._adjuster.update_GRs_target(GRs_target.item())

    def adjust_resources(self):
        new_world_size = torch.tensor(0).to(local_rank())
        if rank() == 0:
            self._conn_to_processor.send('alloc', self._adjuster.ideal_devices)
            cmd, data = self._conn_to_processor.recv()
            assert cmd == 'alloc'
            new_world_size = torch.tensor(int(data)).to(local_rank())
        dist.broadcast(new_world_size, src=0)
        if new_world_size == world_size():
            accum_step, local_bs = self._adjuster.adjust_local_bs()
            self._counter.set_accum(accum_step)
            self._trainloader.set_new_bs(local_bs)
        else:
            if rank() == 0:
                self.save_checkpoint()
            dist.barrier()  # Ensure exiting after checkpoint was saved
            self.exit()
    
    def exit(self):
        self._conn_to_processor.close()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)


    @abstractmethod
    def get_input(self, device, data: Any) -> Any:
        pass

    @abstractmethod
    def get_loss(self, device, data: Any, output: Any) -> Any:
        pass

    @abstractmethod
    def evaluate(self, device, epoch: int, model: torch.nn.Module, testloader: DataLoader) -> Any:
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_start(self, index):
        pass

    def on_batch_end(self, index, data, input, output, loss):
        pass

    def on_forward_start(self, index, data, input):
        pass

    def on_forward_end(self, index, data, input, output):
        pass

    def on_backward_start(self, index, data, input, output, loss):
        pass

    def on_backward_end(self, index, data, input, output, loss):
        pass

    def on_step_start(self, index, data, input, output, loss):
        pass

    def on_step_end(self, index, data, input, output, loss):
        pass
    