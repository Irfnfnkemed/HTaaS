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
import time

from FTaaS.intra.elements import DataLoaderArguments
from FTaaS.intra.job import IntraOptim
import FTaaS.intra.env as env

class yolo_job_optim(IntraOptim):
    
    def __init__(self, model, trainloader_args, testloader, optimizer, epochs, args):
        super().__init__(model, trainloader_args, testloader, optimizer, epochs)
        self.loss_func = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.args = args
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
    
    def get_input(self, device, data):
        imgs = data[0].to(device)
        label_sbbox = data[1].to(device)
        label_mbbox = data[2].to(device)
        label_lbbox = data[3].to(device)
        sbboxes = data[4].to(device)
        mbboxes = data[5].to(device)
        lbboxes = data[6].to(device)
        return imgs
        return (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
    
    def get_loss(self, device, data, output):
        loss, loss_giou, loss_conf, loss_cls = self.loss_func(output[0], output[1], data[1].to(device), data[2].to(device),
                                                    data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device))
        self.loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])        
        return loss
    
    def evaluate(self, device, epoch, model, testloader):
        print(f"[RANK{dist.get_rank()}]: beg-eval {time.time()}")
        mAP = 0
        if epoch == 49:
            with torch.no_grad():
                APs = Evaluator(self.model.module).APs_voc()
                for i in APs:
                    print("{} --> mAP : {}".format(i, APs[i]))
                    mAP += APs[i]
                mAP = mAP / self._trainloader_args.dataset.num_classes
                print('mAP:%g'%(mAP))

    def on_epoch_start(self, epoch):
        print(f"[RANK{dist.get_rank()}]: beg-train {time.time()}")
        self.mloss = torch.zeros(4)
        
    def on_batch_end(self, index, data, input, output, loss):
        self.mloss = (self.mloss * index + self.loss_items) / (index + 1)
        if index%10==0:
            s = ('Batch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                    'lr: %g') % (index, len(self.trainloader) - 1, self.mloss[0], self.mloss[1], self.mloss[2], self.mloss[3],
                                self.optimizer.param_groups[0]['lr'])
        if self.multi_scale_train and (index+1)%10 == 0:
            self._trainloader_args.dataset.img_size = random.choice(range(10,20)) * 32
            print("multi_scale_img_size : {}".format(self._trainloader_args.dataset.img_size))
    

if __name__ == '__main__':       

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true',default=False,  help='resume training flag')
    opt = parser.parse_args()

    epochs = cfg.TRAIN["EPOCHS"]
    weight_path = opt.weight_path
    multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
    train_dataset = data.VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    
    train_args = DataLoaderArguments(train_dataset, num_workers=cfg.TRAIN["NUMBER_WORKERS"], shuffle=True, drop_last=True)

    yolov3 = Yolov3().to(env.local_rank())
    # self.yolov3.apply(tools.weights_init_normal)

    optimizer = optim.SGD(yolov3.parameters(), lr=1e-7, momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
    #self.optimizer = optim.Adam(self.yolov3.parameters(), lr = lr_init, weight_decay=0.9995)

    yolov3.load_darknet_weights(weight_path)

    # scheduler = cosine_lr_scheduler.CosineDecayLR(optimizer,
    #                                                     T_max=epochs*len(train_dataset),
    #                                                     lr_init=1e-7,
    #                                                     lr_min=cfg.TRAIN["LR_END"] / cfg.TRAIN["ACCUMULATE"],
    #                                                     warmup=cfg.TRAIN["WARMUP_EPOCHS"]*len(self.train_dataloader))


    ########################### TRAINING #####################################
    
    job = yolo_job_optim(yolov3, train_args, None, optimizer, epochs, opt)
    job.load_checkpoint()
    job.run()
    