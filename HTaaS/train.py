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
from .monitor import GradMonitor
from .adjuster import Adjuster
from .counter import Counter
from .comm import *
from .env import *
from .config import *


class trainer(ABC):
    
    model: DDP
    trainloader: ElasticDataLoader
    testloader: DataLoader
    optimizer: optim.Optimizer
    grad_monitor: GradMonitor
    adjuster: Adjuster
    counter: Counter
    conn_to_processor: ClientInstance
    total_epochs: int
    now_epochs: int
    job_id: int
    gpu_mem_utilization: float
    
    
     

    def __init__(self, model: torch.nn.Module, trainset: Dataset, 
                 trainloader_args: Dict[str, Any], testloader: DataLoader,
                 optimizer: Optional[torch.optim.Optimizer], total_epochs: int, 
                 gpu_mem_utilization: float = 0.85):
        if not dist.is_available() or not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=world_size(), rank=rank())
        # initialize the trainer
        self.model = DDP(model, device_ids=[local_rank()], output_device=local_rank())
        self.trainloader = ElasticDataLoader(trainset, **trainloader_args)
        self.testloader = testloader
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters())
        else:
            self.optimizer = optimizer
        self.grad_monitor = GradMonitor(self.model, self.trainloader)
        self.adjuster = Adjuster(self.optimizer)
        self.counter = Counter()
        self.conn_to_processor = ClientInstance()
        self.total_epochs = total_epochs
        self.now_epochs = 0
        self.gpu_mem_utilization = gpu_mem_utilization

        # connect to processor and get job-id
        tmp_buffer = torch.tensor(0).to(local_rank())
        if rank() == 0:
            #print(processor_host(), processor_port())
            self.conn_to_processor.connect(processor_host(), processor_port())
            self.conn_to_processor.send('init', '')
            cmd, job_id = self.conn_to_processor.recv()
            assert cmd == 'init'
            tmp_buffer = torch.tensor(int(job_id)).to(local_rank())
        dist.broadcast(tmp_buffer, src=0)
        dist.barrier()
        self.job_id = tmp_buffer.item()

    def save_checkpoint(self):
        checkpoint = {
            'model': self._model.module.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'trainloader_args': self._trainloader_args.save_state(),
            'epochs': self._epochs,
            'now_epochs': self._now_epochs,
            'max_bs': self._adjuster.max_bs,
            'max_global_bs': self._adjuster.max_global_bs,
            'ideal_global_bs': self._ideal_global_bs,
            'accumulation_steps': self._accumulation_steps,
            'adapt': self._adapt,
        }
        torch.save(checkpoint, f'/home/guanjie/HTaaS/launch/tmp_{self._job_id}/checkpoint.pth')

    def load_checkpoint(self):
        if os.path.exists(f'/home/guanjie/HTaaS/launch/tmp_{self._job_id}/checkpoint.pth'):
            print("Loading from checkpoint")
            checkpoint = torch.load(f'/home/guanjie/HTaaS/launch/tmp_{self._job_id}/checkpoint.pth', weights_only=True)
            self._adapt = checkpoint['adapt']
            if self._adapt:
                self._optimizer = self._optimizer_adapt
                self._optimizer_adapt = None
            self._epochs = checkpoint['epochs']
            self._model.module.load_state_dict(checkpoint['model'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._trainloader_args.load_state(checkpoint['trainloader_args'])
            self._now_epochs = checkpoint['now_epochs']
            self._ideal_global_bs = checkpoint['ideal_global_bs']
            self._accumulation_steps = checkpoint['accumulation_steps']
            self._adjuster.set_accumulate_steps(self._accumulation_steps)
            self._adjuster.set_init_bs_config(checkpoint['max_bs'], checkpoint['max_global_bs'])
            self._grad_monitor.set_accumulation_steps(self._accumulation_steps)


            
        else:
            self.profile_max_bs()

    def profile(self):
        max_mem = torch.cuda.get_device_properties(self.model.device).total_memory * self.gpu_mem_utilization
        assert world_size() == 1
        self.model.train()
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
                self.trainloader.set_new_bs(bs)
                with self.trainloader.gen_static_data as datas_grs:
                    tmp_counter = Counter()
                    for _, data_grs in datas_grs:
                        tmp_counter.step()
                        input_grs = self.get_input(local_rank(), data_grs)
                        output_grs = self.model(input_grs)
                        loss_grs = self.get_loss(local_rank(), data_grs, output_grs) / tmp_counter.accum
                        loss_grs.backward()
                        if tmp_counter.need_update:
                            self.grad_monitor.monitor_GRs(True)
                            self.optimizer.zero_grad()
                            if tmp_counter.iter_count == GRS_ACC_TIMES:
                                break # end monitoring GRs
                GRs_list.append(self.grad_monitor.GRs)
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
        
        max_log_bs = np.log2((max_mem - intercept_mem) / slope_mem)
        max_local_bs = 2 ** np.ceil(max_log_bs)
        
        best_global_bs = int(2 ** (GRs_target - best_intercept_GRs) / best_slope_GRs)
        
        
        
    def run(self):
        self.on_train_start()
        while self._now_epochs < self._epochs:
            # print(f'[RANK{rank()}]: bs:{self._ideal_global_bs}, accumu: {self._accumulation_steps}')
            # print(f"[RANK{rank()}]: epoch-beg: {time.time()}")
            
            # training
            self.on_epoch_start(self._now_epochs)
            self.model.train()
            for index, data in enumerate(self.trainloader):
                self.on_batch_start(index)
                self.counter.step()
                input = self.get_input(local_rank(), data)
                self.on_forward_start(index, data, input)
                output = self.model(input)
                self.on_forward_end(index, data, input, output)
                loss = self.get_loss(local_rank(), data, output) / self.counter.accum
                self.on_backward_start(index, data, input, output, loss)
                loss.backward()
                self.on_backward_end(index, data, input, output, loss)
                if self.counter.need_update:
                    # monitor GRt
                    self.grad_monitor.monitor_GRt()
                    # adjust lr according to GRt
                    self.adjuster.adjust_lr()
                    self.on_step_start(index, data, input, output, loss)
                    self.optimizer.step()
                    self.on_step_end(index, data, input, output, loss)
                    self.optimizer.zero_grad()
                    if self.counter.iter_count % GRS_MONITOR_INTERVAL == 0:
                        with self.grad_monitor.env_GRs():
                            # leverage the monitored grad
                            tmp_counter.step()
                            self.grad_monitor.monitor_GRs(False)
                            self.optimizer.zero_grad()
                            # monitor GRs without para update
                            tmp_counter = Counter()
                            tmp_counter.set_accum(self.counter.accum)
                            with self.trainloader.gen_static_data as datas_grs:
                                for _, data_grs in datas_grs:
                                    tmp_counter.step()
                                    input_grs = self.get_input(local_rank(), data_grs)
                                    output_grs = self.model(input_grs)
                                    loss_grs = self.get_loss(local_rank(), data_grs, output_grs) / tmp_counter.accum
                                    loss_grs.backward()
                                    if tmp_counter.need_update:
                                        self.grad_monitor.monitor_GRs(True)
                                        self.optimizer.zero_grad()
                                        if tmp_counter.iter_count == GRS_ACC_TIMES:
                                            break # end monitoring GRs
                self.on_batch_end(index, data, input, output, loss)
                
            print(f"[RANK{rank()}]: epoch-train-finish: {time.time()}")

            # evaluating
            self.model.eval()
            self.evaluate(local_rank(), self._now_epochs, self.model.module, self.testloader)

            print(f"[RANK{rank()}]: epoch-eval-finish: {time.time()}")
            
            # resources adjustment
            self.update_epb_standard()
            
            # Adjust bs according to epb
            self._ideal_global_bs = self._adjuster.adjust_bs(self._ideal_global_bs)
            self.adjust_resources()

            self.on_epoch_end(self._now_epochs)
            self._now_epochs += 1

        self.on_train_end()
        if rank() == 0:
            self.conn_to_processor.send('end', '')
            self.conn_to_processor.close()
        dist.barrier()
        self.exit()

    def update_epb_standard(self):
        # get epb-standard maintained by cluster-sched
        epb_standard = torch.tensor(1.0).to(local_rank())
        if rank() == 0:
            self._ipc.send('status', '')
            cmd, standard = self._ipc.recv()
            assert cmd == 'status'
            epb_standard = torch.tensor(float(standard)).to(local_rank())
        dist.broadcast(epb_standard, src=0)
        epb_standard = epb_standard.item()
        ##############################
        epb_standard = 0.0
        self._adjuster.set_bs_config(epb_standard)

    def adjust_resources(self):
        # Get new_world_size according to cluster-status and ideal_global_bs
        job_status = 0  # 0 for normal, 1 for busy
        init_new_world_size = math.ceil(self._ideal_global_bs / self._adjuster.max_bs)
        new_world_size = max(2, init_new_world_size)
        if new_world_size > world_size():
            alloc_size = torch.tensor(0).to(local_rank())
            if rank() == 0:  # Request new resource-allocation
                self._ipc.send('alloc', new_world_size)
                cmd, data = self._ipc.recv()
                assert cmd == 'alloc'
                alloc_size = torch.tensor(int(data)).to(local_rank())
            dist.broadcast(alloc_size, src=0)
            alloc_size = int(alloc_size.item())
            assert alloc_size >= world_size()
            job_status = 1 if (new_world_size > alloc_size and init_new_world_size > 1) else 0
            new_world_size = alloc_size

        # Resources reallocation
        self.set_accumulation_steps(new_world_size)
        
        if new_world_size > world_size():
            self._now_epochs += 1
            if rank() == 0:
                self.save_checkpoint()
            dist.barrier()  # Ensure exiting after checkpoint was saved
            self.exit()
            return
        elif new_world_size < world_size():
            if rank() == 0:
                self._ipc.send('free', new_world_size)
            self._now_epochs += 1
            if rank() == 0:
                self.save_checkpoint()
            dist.barrier()  # Ensure exiting after checkpoint was saved
            self.exit()
            return
        else:
            if rank() == 0:
                self._ipc.send('heartbeat', job_status)
            dist.barrier()

    def set_accumulation_steps(self, new_world_size: int):
        new_accumulation_steps = math.ceil(self._ideal_global_bs / new_world_size / self._adjuster.max_bs)
        if new_world_size == 1:
            new_accumulation_steps = max(2, new_accumulation_steps)
        self._accumulation_steps = new_accumulation_steps
        self._adjuster.adjust_accumulate_step(new_accumulation_steps)
        self._grad_monitor.set_accumulation_steps(new_accumulation_steps)

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

    def exit(self):
        self.conn_to_processor.close()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)