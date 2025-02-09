import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12153'
import time
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from models import *
import pandas as pd
import json

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument('--lr', default=1e-7, type=float, help='learning rate')
parser.add_argument('--epochs', default=60, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()

class GradMonitor:

    def __init__(self):
        self._model: DDP = None

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

    def associate(self, model: DDP):
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

    def _correct_epb(self, epb_list: List[float]) -> torch.Tensor | None:
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
    def epb(self) -> (torch.Tensor | None):
        if self.epb_average is None:
            return None
        return torch.log2(self.epb_average - 1)
    
class Adjuster:

    def __init__(self):
        self._optimizer: torch.optim.Optimizer = None
        self._grad_monitor: GradMonitor = None
        self.adapt = False

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
            # print(ept)
            if self.low_error <= ept - self.standard_ept <= self.high_error or \
                    math.isnan(rate) or math.isinf(rate):
                rate = 1.0
                self.fit_lr_steps += 1
            else:
                self.fit_lr_steps = max(0, self.fit_lr_steps - 1)
            for param_group in self._optimizer.param_groups:
                print(param_group['lr'])
                param_group['lr'] *= rate
            # print("!!!!!!!!!!!", self.fit_lr_steps)
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
            new_bs = min(int(new_bs / 0.7), bs)
        else:
            new_bs = int(2 ** (epb - self.upper_bound - self.bias) * bs)
            new_bs = max(int(new_bs * 0.7), bs)
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
        return self.accumulation_step if dist.get_world_size() == 1 else dist.get_world_size()
    
    def set_init_bs_config(self, max_bs: int, max_global_bs: int):
        self.max_bs = max_bs
        self.max_global_bs = max_global_bs

    
class Timer:
    def __init__(self):
        self.total = 0
        self.cur = 0
    
    def beg(self):
        self.cur = time.time()
        
    def end(self):
        self.total += time.time() - self.cur
        
    def get(self):
        return self.total


def train(model, device, train_loader, optimizer, criterion, epoch, monitor: GradMonitor, adjuster: Adjuster, timer: Timer, world_size):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        with model.no_sync():
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            timer.beg()
            if (batch_idx + 1) % 5 == 0: 
                monitor.monitor_epb(True)
            timer.end()
        if world_size > 1:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad /= world_size
        timer.beg()
        monitor.monitor_ept()
        adjuster.adjust_lr()
        timer.end()
        # x_index = epoch + batch_idx / len(train_loader)
        # with open(f'ep/ept_{args.lr}_{args.bs}_{world_size}.txt', 'a') as file:
        #     file.write(f'{x_index} {monitor.ept}\n')
        # with open(f'ep/epb_{args.lr}_{args.bs}_{world_size}.txt', 'a') as file:
        #     file.write(f'{x_index} {monitor.epb}\n')
        optimizer.step()      
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        if batch_idx % 100 == 99:
            print(f'[Rank{dist.get_rank()}] Epoch: {epoch}, Batch: {batch_idx}, Loss: {running_loss / 100:.6f}')
            running_loss = 0.0       
    print(f'Accuracy: {100 * correct / total}%')
  
def test(model, device, test_loader, criterion, world_size, monitor, result):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print(f'Test set: Average loss: {test_loss / len(test_loader.dataset):.4f}, Accuracy: {100 * correct / total}%')
        # with open(f'ep/acc_{args.lr}_{args.bs}_{world_size}.txt', 'a') as file:
        #     file.write(str(100 * correct / total) + "\n")
        # with open(f'ep/loss_{args.lr}_{args.bs}_{world_size}.txt', 'a') as file:
        #     file.write(str(test_loss / len(test_loader.dataset)) + "\n")     
    result['time'].append(time.time())
    result['GRT'].append(monitor.ept.item())      
    result['GRS'].append(monitor.epb.item() - 2)
    result['valid'].append(100 * correct / total)                     
    result['loss'].append(test_loss / len(test_loader.dataset))        

def main(rank, world_size):
    
    ALL_BEG_TIME = time.time()
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_sampler = DistributedSampler(trainset)
    # train_loader = DataLoader(trainset, batch_size=args.bs//world_size, shuffle=False,num_workers=2, pin_memory=True, sampler=train_sampler)
   
    if rank == 0:
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = eval(args.model)()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=5e-4)

    # lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)

    monitor = GradMonitor()
    adjuster = Adjuster()
    monitor.associate(model)
    adjuster.associate(optimizer, monitor)
    
    
    
    # profile
    model.train()
    cur_bs = 4
    epb_list = []
    while True:
        try:
            train_loader = DataLoader(trainset, batch_size=cur_bs//world_size, shuffle=False,
                                      num_workers=2, pin_memory=True, sampler=train_sampler)
            if len(train_loader) <= 20:
                break
            sample_epb = []
            
            for index, (data, target) in enumerate(train_loader):
                with model.no_sync():
                    data, target = data.to(rank), target.to(rank)
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    monitor.monitor_epb(True)
                    epb = monitor.epb
                    optimizer.zero_grad()
                    if epb is not None:
                        sample_epb.append(epb)
                    if len(sample_epb) >= 3:
                        sample_epb = np.array(sample_epb)
                        # q1 = np.percentile(sample_epb, 25)
                        # q3 = np.percentile(sample_epb, 75)
                        # iqr = q3 - q1
                        # lower_bound = q1 - 1.5 * iqr
                        # upper_bound = q3 + 1.5 * iqr
                        # sample_epb = sample_epb[(sample_epb >= lower_bound) & (sample_epb <= upper_bound)]
                        epb = sample_epb.mean()
                        epb_list.append(epb)
                        cur_bs *= 2
                        monitor.epb_average = None
                        break
                if (index + 1) >= 10:
                    epb_list.append(-1)
                    cur_bs *= 2
                    monitor.epb_average = None
                    break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
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
    
    adjuster.set_bs_config(0)
    ideal = (adjuster.lower_bound + adjuster.upper_bound) / 2 + adjuster.bias
    ideal_global_bs = math.ceil(2 ** ((ideal - b) / k))
    adjuster.set_init_bs_config(math.ceil(cur_bs/world_size/2), ideal_global_bs * 16)
    # profile ended
    
    TRAIN_BEG_TIME = time.time()
    
    result = {"time": [], "GRT": [], "GRS": [], "valid": [], "loss": []}
    
    timer_monitor = Timer()
    timer_eval = Timer()
    
    bs = ideal_global_bs
    for epoch in range(0, args.epochs):   
        print("bs:", bs)
        train_loader = DataLoader(trainset, batch_size=bs//world_size, shuffle=False,
                                      num_workers=2, pin_memory=True, sampler=train_sampler)
        train(model, rank, train_loader, optimizer, criterion, epoch, monitor, adjuster, timer_monitor, world_size)
        timer_eval.beg()
        if rank == 0:
            test(model.module, rank, test_loader, criterion, world_size, monitor, result)
        timer_eval.end()
        timer_monitor.beg()
        if monitor.stop_ept and not adjuster.adapt:
            lr_list = []
            for param_group in optimizer.param_groups:
                lr_list.append(param_group['lr'])
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
            for i, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_list[i] * 0.5
            print(lr_list)
            adjuster.associate(optimizer, monitor)
            adjuster.adapt = True
        bs = adjuster.adjust_bs(bs)
        print(bs)
        # lr_scheduler.step()
        timer_monitor.end()
        
    result['time'] = [x - ALL_BEG_TIME for x in result['time']]
    
    df = pd.DataFrame(result)
    df.to_csv(f'info/{args.lr}_{args.bs}.csv', index=False)
    
    data = {
        'preprocessing_time': TRAIN_BEG_TIME - ALL_BEG_TIME,
        'monitor_time': timer_monitor.get(),
        'eval_time': timer_eval.get(),
    }
    with open(f'overhead/{args.lr}_{args.bs}.json', 'w') as file:
        json.dump(data, file)
       
        


if __name__ == "__main__":
    world_size = 4
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)