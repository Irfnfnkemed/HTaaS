from typing import List, Dict
from torch.utils.data import DataLoader, DistributedSampler
from model.yolov3 import Yolov3
from model.loss.yolo_loss import YoloV3Loss
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import utils.datasets as data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import random
import argparse
from eval.evaluator import *
from utils.tools import *
import config.yolov3_config_voc as cfg
from utils import cosine_lr_scheduler
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12339'

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
        
        self.t = 0

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

    def set_accumulation_steps(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps

    def _monitor_grad(self) -> torch.Tensor:
        para_grad = torch.zeros(self.grad_dim).to(dist.get_rank())
        cur_index = 0
        for index, param in enumerate(self._model.parameters()):
            if len(self.grad_mask) > 0:
                if param.grad is not None:
                    para_grad[cur_index: cur_index + self.grad_mask_len[index]] = param.grad.view(-1)[self.grad_mask[index]]
                cur_index += self.grad_mask_len[index]
            else:
                num_ele = param.data.numel()
                if param.grad is not None:
                    para_grad[cur_index: cur_index + num_ele] = param.grad.view(-1)
                cur_index += num_ele
        return para_grad

    def monitor_ept(self):
        self.t += 1
        if self.t > 100:
            return
        para_grad = self._monitor_grad()
        self.mean_ept = self.gamma * self.mean_ept + (1 - self.gamma) * para_grad
        self.variance_ept = self.gamma * self.variance_ept + (1 - self.gamma) * (para_grad ** 2)
        ept = self._ept()
        self.ept_average = self.ept_average * self.ept_ema + ept * (1 - self.ept_ema)

    def monitor_epb(self, flag: bool):
        if self.t > 100:
            return
        if dist.get_world_size() > 1 and flag:
            para_grad = self._monitor_grad()
            from_rank = dist.get_rank() - 1
            to_rank = dist.get_rank() + 1
            self.mean_epb = torch.tensor(0.0)
            self.variance_epb = torch.tensor(0.0)

            mean = torch.empty_like(para_grad).to(dist.get_rank())
            variance = torch.empty_like(para_grad).to(dist.get_rank())

            if from_rank >= 0:
                result = dist.recv(mean, src=from_rank, tag=1)
                assert result == from_rank
                result = dist.recv(variance, src=from_rank, tag=2)
                assert result == from_rank

            if dist.get_rank() == 0:
                self.mean_epb = para_grad
                self.variance_epb = para_grad ** 2
            else:
                self.mean_epb = mean + para_grad
                self.variance_epb = variance + para_grad ** 2

            if to_rank < dist.get_world_size():
                mean = self.mean_epb.clone().detach().to(dist.get_rank())
                variance = self.variance_epb.clone().detach().to(dist.get_rank())
                dist.send(mean, dst=to_rank, tag=1)
                dist.send(variance, dst=to_rank, tag=2)

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
            para_grad = self._monitor_grad()
            self.now_steps += 1
            self.variance_epb = self.variance_epb + (para_grad - self.mean_epb) ** 2
            self.mean_epb = para_grad
            self.epb_list.append(self._epb())
            if self.now_steps == self.accumulation_steps:
                self.now_steps = 0
                self.mean_epb = torch.tensor(0.0)
                self.variance_epb = torch.tensor(0.0)
                print(self.epb_list)
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
        ratio = ratio[ratio < 1]
        ratio = ratio[ratio > 0]
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
    def epb(self) -> torch.Tensor | None:
        if self.epb_average is None:
            return None
        return torch.log2(self.epb_average - 1)


    
class Trainer(object):
    def __init__(self,  weight_path, resume):
        self.lr_init = cfg.TRAIN["LR_INIT"] / cfg.TRAIN["ACCUMULATE"]
        init_seeds(0)
        self.device = dist.get_rank()
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        train_sampler = DistributedSampler(self.train_dataset)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=cfg.TRAIN["BATCH_SIZE"], shuffle=False, num_workers=cfg.TRAIN["NUMBER_WORKERS"], sampler=train_sampler)
        
        self.yolov3 = Yolov3().to(dist.get_rank())
        self.yolov3 = DDP(self.yolov3, device_ids=[dist.get_rank()], output_device=dist.get_rank())
        # self.yolov3.apply(tools.weights_init_normal)

        self.optimizer = optim.SGD(self.yolov3.parameters(), lr=self.lr_init,
                                   momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=self.lr_init,
                                                          lr_min=cfg.TRAIN["LR_END"] / cfg.TRAIN["ACCUMULATE"],
                                                          warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))
        self.monitor = GradMonitor()
        self.monitor.associate(self.yolov3)



    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(os.path.split(weight_path)[0], "last.pt")
            chkpt = torch.load(last_weight, map_location=dist.get_rank())
            self.yolov3.module.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            self.yolov3.module.load_darknet_weights(weight_path)


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], f'backup_epoch{epoch}_{self.lr_init}_{cfg.TRAIN["BATCH_SIZE"]}_{cfg.TRAIN["ACCUMULATE"]}_{dist.get_world_size()}.pt'))
        del chkpt


    def train(self):
        print(self.yolov3)
        print("Train datasets number is : {}".format(len(self.train_dataset)))
        
        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(self.train_dataloader):
                if (i + 1) % cfg.TRAIN['ACCUMULATE'] == 0:
                    self.scheduler.step(len(self.train_dataloader)*epoch + i)
                with self.yolov3.no_sync():
                    imgs = imgs.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    sbboxes = sbboxes.to(self.device)
                    mbboxes = mbboxes.to(self.device)
                    lbboxes = lbboxes.to(self.device)

                    p, p_d = self.yolov3(imgs)

                    loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                    label_lbbox, sbboxes, mbboxes, lbboxes)
                    loss.backward()
            
                if (i+1) % cfg.TRAIN['ACCUMULATE'] == 0:
                    self.monitor.monitor_epb(True)
                    if dist.get_world_size() > 1:
                        for param in self.yolov3.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                                param.grad /= dist.get_world_size()
                    self.monitor.monitor_ept()
                    
                    if self.monitor.t < 100:
                        if dist.get_rank() == 0:
                            with open(f'ep/ept_{self.lr_init}_{cfg.TRAIN["BATCH_SIZE"]}_{cfg.TRAIN["ACCUMULATE"]}_{dist.get_world_size()}.log', 'a') as file:
                                file.write(f"{self.monitor.ept}\n")
                            with open(f'ep/epb_{self.lr_init}_{cfg.TRAIN["BATCH_SIZE"]}_{cfg.TRAIN["ACCUMULATE"]}_{dist.get_world_size()}.log', 'a') as file: 
                                file.write(f"{self.monitor.epb}\n")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # Print batch results
                if i%10==0:
                    s = ('Epoch:[ %d | %d ]    Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                         'lr: %g') % (epoch, self.epochs - 1, i, len(self.train_dataloader) - 1, mloss[0],mloss[1], mloss[2], mloss[3],
                                      self.optimizer.param_groups[0]['lr'])
                    if dist.get_rank() == 0:
                        print(s)
                        with open(f'ep/loss_{self.lr_init}_{cfg.TRAIN["BATCH_SIZE"]}_{cfg.TRAIN["ACCUMULATE"]}_{dist.get_world_size()}.log', 'a') as file:
                            file.write(f"{mloss[3].item()}\n")

                # multi-sclae training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size))

            mAP = 0
            if epoch == 49:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            print('best mAP : %g' % (self.best_mAP))
        self.__save_model_weights(50, mAP)


def main(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    opt = parser.parse_args()
     
    Trainer(weight_path=opt.weight_path, resume=opt.resume).train()
    
if __name__ == "__main__":
    world_size = 2
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)