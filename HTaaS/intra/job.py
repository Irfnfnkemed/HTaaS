import math
import sys
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from .elements import GradMonitor, Adjuster, DataLoaderArguments
from .env import *
from ..inter.ipc import ClientIPC


class IntraOptim(ABC):

    def __init__(self, model: torch.nn.Module, trainloader_args: DataLoaderArguments,
                 testloader: DataLoader, optimizer: torch.optim.Optimizer, optimizer_adapt: torch.optim.Optimizer, epochs: int):
        if not dist.is_available() or not dist.is_initialized():
            dist.init_process_group(backend='nccl', world_size=world_size(), rank=rank())
        self._model = DDP(model, device_ids=[local_rank()], output_device=local_rank())
        self._trainloader_args = trainloader_args
        self._testloader = testloader
        self._optimizer = optimizer
        self._optimizer_adapt = optimizer_adapt
        self._epochs = epochs
        self._ideal_global_bs = 0
        self._accumulation_steps = 1
        self._grad_monitor = GradMonitor()
        self._adjuster = Adjuster()
        self._ipc = ClientIPC()
        self._now_epochs = 0
        self._trainloader = None
        self._adapt = False

        self._grad_monitor.associate(self._model)
        self._adjuster.associate(self._optimizer, self._grad_monitor)

        # Connect to proxy-job and get job-id
        self._job_id = torch.tensor(0).to(local_rank())
        if rank() == 0:
            print(proxy_host(), proxy_port())
            self._ipc.connect(proxy_host(), proxy_port())
            self._ipc.send('init', '')
            cmd, job_id = self._ipc.recv()
            assert cmd == 'init'
            self._job_id = torch.tensor(int(job_id)).to(local_rank())
        dist.broadcast(self._job_id, src=0)
        dist.barrier()
        self._job_id = self._job_id.item()

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

    def profile_max_bs(self):
        assert world_size() == 1
        self._model.train()
        ideal_global_bs = 4
        accumu_steps = 4
        self._adjuster.set_accumulate_steps(accumu_steps)
        self._grad_monitor.set_accumulation_steps(accumu_steps)
        epb_list = []
        while True:
            try:
                loader_args = self._trainloader_args.get_args()
                loader_args['shuffle'] = self._trainloader_args.shuffle
                bs = math.ceil(ideal_global_bs / accumu_steps) 
                dataloader = DataLoader(batch_size=bs, **loader_args)
                if len(dataloader) <= 5 * accumu_steps:
                    break
                sample_epb = []
                for index, data in enumerate(dataloader):
                    input = self.get_input(local_rank(), data)
                    output = self._model(input)
                    loss = self.get_loss(local_rank(), data, output)
                    loss.backward()
                    self._grad_monitor.monitor_epb(False)
                    if (index + 1) % accumu_steps == 0:
                        epb = self._grad_monitor.epb
                        self._model.zero_grad()
                        if epb is not None:
                            sample_epb.append(epb)
                        if len(sample_epb) >= 3:
                            sample_epb = np.array(sample_epb)
                            q1 = np.percentile(sample_epb, 25)
                            q3 = np.percentile(sample_epb, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            sample_epb = sample_epb[(sample_epb >= lower_bound) & (sample_epb <= upper_bound)]
                            epb = sample_epb.mean()
                            epb_list.append(epb)
                            ideal_global_bs *= 2
                            self._grad_monitor.epb_average = None
                            break
                    if (index + 1) >= accumu_steps * 10:
                        epb_list.append(-1)
                        ideal_global_bs *= 2
                        self._grad_monitor.epb_average = None
                        break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("AAAAAAAAAAAAAAAAAAAAAAAAAAA")
                    break
                else:
                    raise
        print(epb_list)
        
        x = np.arange(0, len(epb_list))
        y = np.array(epb_list)
        condition = (y >= -5) & (y <= 5)
        max_start = 0
        max_end = 0
        current_start = 0
        max_length = 0
        for i in range(len(condition)):
            if condition[i]:
                if not condition[i - 1] if i > 0 else True:
                    current_start = i
                current_length = i - current_start + 1
                if current_length > max_length:
                    max_length = current_length
                    max_start = current_start
                    max_end = i
            else:
                current_start = i + 1
        condition = np.full_like(condition, False)
        condition[max_start:max_end + 1] = True
        x = x[condition]
        y = y[condition]
        
        k = ((x * y).mean() - x.mean() * y.mean()) / ((x ** 2).mean() - (x.mean()) ** 2)
        b = y.mean() - k * x.mean()
        print(f"k={k},b={b}", math.ceil(ideal_global_bs / accumu_steps / 2))
        
        
        self.update_epb_standard()
        ideal = (self._adjuster.lower_bound + self._adjuster.upper_bound) / 2 + self._adjuster.bias
        self._ideal_global_bs = math.ceil(2 ** ((ideal - b) / k))
        self._adjuster.set_init_bs_config(math.ceil(ideal_global_bs / accumu_steps / 2), self._ideal_global_bs * 32)
        self.adjust_resources()
        
    def get_dataloader(self):
        sampler = DistributedSampler(self._trainloader_args.dataset, num_replicas=world_size(),
                                        rank=rank(), shuffle=self._trainloader_args.shuffle)
        bs = math.ceil(self._ideal_global_bs / world_size() / self._accumulation_steps)
        self._trainloader = DataLoader(batch_size=bs, sampler=sampler, **self._trainloader_args.get_args())
        sampler.set_epoch(self._now_epochs)
            
        
    def run(self):
        self.on_train_start()
        while self._now_epochs < self._epochs:
            print(f'[RANK{rank()}]: bs:{self._ideal_global_bs}, accumu: {self._accumulation_steps}')
            print(f"[RANK{rank()}]: epoch-beg: {time.time()}")
            
            # set trainloader
            self.get_dataloader()
            
            self.on_epoch_start(self._now_epochs)

            # training
            self._model.train()
            for index, data in enumerate(self._trainloader):
                self.train_epoch(index, data)
                
            print(f"[RANK{rank()}]: epoch-train-finish: {time.time()}")

            # evaluating
            self._model.eval()
            self.evaluate(local_rank(), self._now_epochs, self._model.module, self._testloader)

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
            self._ipc.send('end', '')
            self._ipc.close()
        dist.barrier()
        self.exit()

    def train_epoch(self, index, data):
        self.on_batch_start(index)

        with self._model.no_sync():
            input = self.get_input(local_rank(), data)
            self.on_forward_start(index, data, input)
            output = self._model(input)
            self.on_forward_end(index, data, input, output)

            loss = self.get_loss(local_rank(), data, output)
            self.on_backward_start(index, data, input, output, loss)
            loss.backward()
            self.on_backward_end(index, data, input, output, loss)
            if (index + 1) % 5 == 0:
                self._grad_monitor.monitor_epb(False)

        if (index + 1) % self._accumulation_steps == 0:
            if (index + 1) % 5 == 0:
                self._grad_monitor.monitor_epb(True)
            if world_size() > 1:
                for param in self._model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= world_size()
            self._grad_monitor.monitor_ept()
            self._adjuster.adjust_lr()
            self.on_step_start(index, data, input, output, loss)
            self._optimizer.step()
            self.on_step_end(index, data, input, output, loss)
            self._optimizer.zero_grad()
            if self._grad_monitor.stop_ept and not self._adapt:
                lr_list = []
                for param_group in self._optimizer.param_groups:
                    lr_list.append(param_group['lr'])
                self._optimizer = self._optimizer_adapt
                for i, param_group in enumerate(self._optimizer.param_groups):
                    param_group['lr'] = lr_list[i]
                self._adjuster.associate(self._optimizer, self._grad_monitor)
                self._optimizer_adapt = None
                self._adapt = True

        self.on_batch_end(index, data, input, output, loss)

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

    @property
    def trainloader(self) -> DataLoader:
        return self._trainloader

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def model(self) -> DDP:
        return self._model

    @property
    def now_epochs(self) -> int:
        return self._now_epochs

    def exit(self):
        self._ipc.close()
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        sys.exit(0)
