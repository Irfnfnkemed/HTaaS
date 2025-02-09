# Code adapted from https://github.com/guoyang9/NCF

import json
import os

import pandas as pd
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12111'
import time
import argparse
from typing import List, Dict, Tuple
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import model
import evaluate
import data_utils
import os.path
import tqdm
import math

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=1e-7,
                    help="learning rate")
parser.add_argument("--dropout",
                    type=float,
                    default=0.1,
                    help="dropout rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=4096,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="1",
                    help="gpu card ID")
parser.add_argument("--autoscale-bsz",
                    dest='autoscale_bsz',
                    default=False,
                    action='store_true',
                    help="Use AdaptDL batchsize autoscaling")
parser.add_argument("--gradient-accumulation",
                    dest='gradient_accumulation',
                    default=False,
                    action='store_true',
                    help="Use AdaptDL batchsize autoscaling")
parser.add_argument("--dataset",
                    type=str,
                    choices=['ml-1m', 'pinterest-20'],
                    default="ml-1m")
parser.add_argument("--model-type",
                    dest="model_type",
                    type=str,
                    choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'],
                    default="NeuMF-end")
parser.add_argument("--port",
                    type=str,
                    default='12345')
args = parser.parse_args()
os.environ['MASTER_PORT'] = args.port

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

        # config of ept
        self.warmup_ept = 20
        self.adjust_interval = 4
        self.low_error = -0.03
        self.high_error = 0.04
        self.eta = 2
        self.standard_ept = 1.35
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


    
    
def main(rank, world_size):
    ALL_BEG_TIME = time.time()
    
    # 设备配置
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


    dataset = args.dataset
    model_type = args.model_type
    # paths
    main_path = "./data"

    train_rating = os.path.join(main_path, '{}.train.rating'.format(dataset))
    test_rating = os.path.join(main_path, '{}.test.rating'.format(dataset))
    test_negative = os.path.join(main_path, '{}.test.negative'.format(dataset))

    model_path = os.path.join(main_path, 'models')
    GMF_model_path = os.path.join(model_path, 'GMF.pth')
    MLP_model_path = os.path.join(model_path, 'MLP.pth')
    NeuMF_model_path = os.path.join(model_path, 'NeuMF.pth')

    ############################## PREPARE DATASET ##########################
    train_data, test_data, user_num, item_num, train_mat = \
        data_utils.load_all(main_path, train_rating, test_negative, dataset)

    # construct the train and test datasets
    train_dataset = data_utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
    test_dataset = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
    
    # 创建分布式Sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
    # 创建支持分布式训练的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size//world_size, sampler=train_sampler, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_num_ng+1, shuffle=False, num_workers=0) 
    
    ########################### CREATE MODEL #################################
    if model_type == 'NeuMF-pre':
        assert os.path.exists(GMF_model_path), 'lack of GMF model'
        assert os.path.exists(MLP_model_path), 'lack of MLP model'
        GMF_model = torch.load(GMF_model_path)
        MLP_model = torch.load(MLP_model_path)
    else:
        GMF_model = None
        MLP_model = None

    network = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                        args.dropout, model_type, GMF_model, MLP_model)

    network = network.to(rank)
    network = torch.nn.parallel.DistributedDataParallel(network, device_ids=[rank], output_device=rank)

    loss_function = torch.nn.BCEWithLogitsLoss()
    monitor = GradMonitor()
    adjuster = Adjuster()
    monitor.associate(network)
    
    
    timer_monitor = Timer()
    timer_eval = Timer()


    if model_type == 'NeuMF-pre':
        optimizer = optim.SGD(network.parameters(), lr=args.lr)
    else:
        #optimizer = optim.SGD(network.parameters(), lr=args.lr)
        optimizer = optim.AdamW(network.parameters(), lr=args.lr)
        
    adjuster.associate(optimizer, monitor)
        
    result = {"time": [], "GRT": [], "GRS": [], "valid": [], "loss": []}
    
    train_loader.dataset.ng_sample()
    
    
    # profile
    network.train()
    cur_bs = 2
    epb_list = []
    while True:
        try:
            train_loader = DataLoader(train_dataset, batch_size=cur_bs//world_size, sampler=train_sampler, num_workers=0, drop_last=True)
            if len(train_loader) <= 20:
                break
            sample_epb = []
                     
            for index, (user, item, label) in enumerate(train_loader):
                with network.no_sync():
                    user = user.to(rank)
                    item = item.to(rank)
                    label = label.float().to(rank)
                    optimizer.zero_grad()
                    prediction = network(user, item)
                    loss = loss_function(prediction, label)
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
    adjuster.set_init_bs_config(int(cur_bs / 2 / world_size), ideal_global_bs * 16)
    
    
        
    ########################### TRAINING #####################################
    TRAIN_BEG_TIME = time.time()
    print(ALL_BEG_TIME-TRAIN_BEG_TIME)
    return
    
    bs = ideal_global_bs
    for epoch in range(args.epochs):
        train_loader = DataLoader(train_dataset, batch_size=bs//world_size, sampler=train_sampler, num_workers=0, drop_last=True)        
        
        print("bs:", bs)
        
        network.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
        for index, (user, item, label) in enumerate(train_loader):
            with network.no_sync():
                user = user.to(rank)
                item = item.to(rank)
                label = label.float().to(rank)
                optimizer.zero_grad()
                prediction = network(user, item)
                loss = loss_function(prediction, label)
                loss.backward()
                
                timer_monitor.beg()
                if (index + 1) % 5 == 0: 
                    monitor.monitor_epb(True)
                timer_monitor.end()
            
            if world_size > 1:
                for param in network.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                        param.grad /= world_size

            timer_monitor.beg()
            monitor.monitor_ept()
            adjuster.adjust_lr()
            timer_monitor.end()
            
            # x_index = epoch + index / len(train_loader)
            # if monitor.t < 100:
            #     with open(f'ep/ept_{args.lr}_{args.batch_size}_{world_size}.txt', 'a') as file:
            #         file.write(f'{x_index} {monitor.ept}\n')
            #     with open(f'ep/epb_{args.lr}_{args.batch_size}_{world_size}.txt', 'a') as file:
            #         file.write(f'{x_index} {monitor.epb}\n')
            
            optimizer.step()
                           
        timer_eval.beg() 
        network.eval()        
        HR, NDCG = evaluate.metrics(network, test_loader, args.top_k)
        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        timer_eval.end() 
        
        # if rank == 0:   
        #     with open(f'ep/HR_{args.lr}_{args.batch_size}_{world_size}.txt', 'a') as file:
        #         file.write(f'{x_index} {np.mean(HR)}\n')
        #     with open(f'ep/NDCG_{args.lr}_{args.batch_size}_{world_size}.txt', 'a') as file:
        #         file.write(f'{x_index} {np.mean(NDCG)}\n') 
        
        result['time'].append(time.time() - TRAIN_BEG_TIME)
        result['GRT'].append(monitor.ept.item())      
        result['GRS'].append(monitor.epb.item() - 1)
        result['valid'].append(np.mean(HR))                     
        result['loss'].append(loss.item())
          
        dist.barrier()
        
        timer_monitor.beg()
        bs = adjuster.adjust_bs(bs)
        # lr_scheduler.step()
        timer_monitor.end()            

    df = pd.DataFrame(result)
    df.to_csv(f'info/{args.lr}_{args.batch_size}.csv', index=False)
    
    data = {
        'preprocessing_time': TRAIN_BEG_TIME - ALL_BEG_TIME,
        'monitor_time': timer_monitor.get(),
        'eval_time': timer_eval.get(),
    }
    with open(f'overhead/{args.lr}_{args.batch_size}.json', 'w') as file:
        json.dump(data, file)



if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)