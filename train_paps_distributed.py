import argparse
import collections

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
import albumentations as A
import albumentations.pytorch
from tqdm.notebook import tqdm

import cv2
import re
import time

import sys
sys.path.append('../')
from retinanet import coco_eval
from retinanet import csv_eval
from retinanet import model
# from retinanet import retina
from retinanet.dataloader import *
from retinanet.anchors import Anchors
from retinanet.losses import *
from retinanet.scheduler import *
from retinanet.parallel import DataParallelModel, DataParallelCriterion

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import Adam, lr_scheduler
import torch.optim as optim

# distributed for multi gpu using
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def main(args=None):
    
    parser = argparse.ArgumentParser(description='Simple paps training script for training a RetinaNet network.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=60)
    parser.add_argument('--batch_size', help='Number of epochs', type=int, default=32)    
    parser.add_argument('--train_data', help='train data file', default='data/train.npy')    
    parser.add_argument('--test_data', help='test data file', default='data/test.npy')  
    parser.add_argument('--world_size', help='number of distributed workers', type=int, default=1)
    parser.add_argument('--distributed ', help='distributed or not', type=bool, default=True)  
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')    
    
    args = parser.parse_args(args)

#     ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = 4
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, 
             args=(ngpus_per_node, args))

def main_worker(gpu, ngpus_per_node, args):

    global best_acc1
    args.gpu = gpu
    torch.cuda.set_device(args.gpu)  
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    print("Use GPU: {} for training".format(args.gpu))
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank', args.rank)

    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:22',
                            world_size=args.world_size, 
                            rank=args.rank)    
    
    PATH_TO_WEIGHTS = 'coco_resnet_50_map_0_335_state_dict.pt'
    pretrained_retinanet = model.resnet50(num_classes=80, device=device)
    pretrained_retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))

    retinanet = model.resnet50(num_classes=5, device=device)
    for param, state in zip(pretrained_retinanet.parameters(), pretrained_retinanet.state_dict()) :
        #print(state)
        if 'classificationModel' not in state :
            retinanet.state_dict()[state] = param

    for param, state in zip(pretrained_retinanet.fpn.parameters(), pretrained_retinanet.fpn.state_dict()) :
        #print(state)
        retinanet.fpn.state_dict()[state] = param

    for param, state in zip(pretrained_retinanet.regressionModel.parameters(), pretrained_retinanet.regressionModel.state_dict()) :
        #print(state)
        retinanet.regressionModel.state_dict()[state] = param   
    
    retinanet.to(device)
    retinanet = DistributedDataParallel(retinanet, device_ids=[args.gpu])
    # retinanet.cuda()
    retinanet.module.freeze_bn()     
     
    train_info = np.load(args.train_data, allow_pickle=True, encoding='latin1').item()
    
    train_ds = PapsDataset(train_info, transforms)
    
    if parser.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None       

    train_data_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True, 
        sampler=train_sampler
    )

    criterion = FocalLoss(device)
    criterion = criterion.to(device)
    retinanet.training = True
    
    optimizer = optim.Adam(retinanet.parameters(), lr = 1e-7)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.0005,  T_up=5, gamma=0.5)    
    
    loss_per_epoch = 2
    for epoch in range(args.epochs) :
        epoch_loss = []
        total_loss = 0
        retinanet.train()
#         tk0 = tqdm(train_data_loader, total=len(train_data_loader, file=sys.stdout), leave=False)
        tk0 = tqdm(train_data_loader, total=len(train_data_loader), leave=False)
        EPOCH_LEARING_RATE = optimizer.param_groups[0]["lr"]
        print("*****{}th epoch, learning rate {}".format(epoch, EPOCH_LEARING_RATE))

        for step, data in enumerate(tk0) :
            images, _, paths, targets = data
            batch_size = len(images)

            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)

            targets = [ t.to(device) for t in targets]

            outputs = retinanet([images, targets])
            classification, regression, anchors, annotations = (outputs)
            classification_loss, regression_loss = criterion(classification, regression, anchors, annotations)
   
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss 
            total_loss += loss.item()

            epoch_loss.append((loss.item()))
            tk0.set_postfix(lr=optimizer.param_groups[0]["lr"], batch_loss=loss.item(), cls_loss=classification_loss.item(), 
                            reg_loss=regression_loss.item(), avg_loss=total_loss/(step+1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.02)
            optimizer.step()   

        print('{}th epochs loss is {}'.format(epoch, np.mean(epoch_loss)))
        if loss_per_epoch > np.mean(epoch_loss):
            print('best model is saved')
            torch.save(retinanet.state_dict(), 'best_model.pt')
            loss_per_epoch = np.mean(epoch_loss)
        scheduler.step()    
        
if __name__ == '__main__':
    main()        