import argparse
import collections

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
import albumentations as A
import albumentations.pytorch
# from tqdm.notebook import tqdm
from tqdm import tqdm

import cv2
import re
import time

import sys
sys.path.append('../')
from retinanet import model
from retinanet import coco_eval
from retinanet import paps_eval
from retinanet import csv_eval

# from retinanet import retina
# from retinanet.paps_eval import evaluate_paps
from retinanet.dataloader import *
from retinanet.anchors import Anchors
from retinanet.losses import *
from retinanet.scheduler import *
from retinanet.parallel import DataParallelModel, DataParallelCriterion

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

#Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import Adam, lr_scheduler
import torch.optim as optim

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

from pycocotools.cocoeval import COCOeval
import json
import torch

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple paps training script for training a RetinaNet network.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=160)
    parser.add_argument('--batch_size', help='Number of batchs', type=int, default=64)    
    parser.add_argument('--train_data', help='train data file', default='data/train.npy')    
    parser.add_argument('--test_data', help='test data file', default='data/test.npy')   
    parser.add_argument('--saved_dir', help='saved dir', default='trained_models/resnet50_320/') 
    parser.add_argument('--gpu_num', help='default gpu', type=int, default=5) 
    parser.add_argument('--freeze_ex_bn', help='default gpu', type=bool, default=False) 
    
    parser = parser.parse_args(args)
    print('batch_size ', parser.batch_size)
    print('epochs ', parser.epochs)

    # GPU 할당 변경하기
    GPU_NUM = parser.gpu_num
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print(device)
    print ('Current cuda device ', torch.cuda.current_device()) # check
    device_ids = [3,2,1]
    
    PATH_TO_WEIGHTS = 'coco_resnet_50_map_0_335_state_dict.pt'
    pre_retinanet = model.resnet50(num_classes=80, device=device)
    pre_retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS, map_location=device), strict=False)
    pre_retinanet.classificationModel.output = nn.Conv2d(256, 9*2, kernel_size=3, padding=1)

    ret_model = model.resnet50(num_classes=2, device=device)
    ret_model.load_state_dict(pre_retinanet.state_dict())
    del pre_retinanet 
    
    state_dict = ret_model.state_dict()
    for s in state_dict:
        if 'bn' in s :
            if 'weight' in s :
                shape = state_dict[s].shape
                state_dict[s] = torch.zeros(shape)
            elif 'bias' in s :
                shape = state_dict[s].shape
                state_dict[s] = torch.ones(shape)            
    
    ret_model.load_state_dict(state_dict)
    
#     if parser.freeze_ex_bn :
#         for k, p in zip(ret_model.module.state_dict(), ret_model.module.parameters()) :
#             if 'bn' not in k :
#                 p.requires_grad = False  
        
    ret_model = torch.nn.DataParallel(ret_model, device_ids = [3,2,1], output_device=GPU_NUM).to(device)
    ret_model.module.freeze_ex_bn(False)
    # ret_model = DataParallelModel(ret_model, device_ids = device_ids)
    ret_model.to(device)
    # ret_model.cuda()
#     ret_model.module.freeze_bn()     
    
    batch_size = parser.batch_size
    dataset_train = PapsDataset('data/', set_name='train_2class',
                                transform=train_transforms)

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    dataset_val = PapsDataset('data/', set_name='val_2class',
                                transform=val_transforms)

    val_data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )        

    criterion = FocalLoss(device)
    criterion = criterion.to(device)
    ret_model.training = True
    
    optimizer = optim.Adam(ret_model.parameters(), lr = 1e-7)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_max=0.0008,  T_up=5, gamma=0.5)    
    
    loss_per_epoch = 0.6
    optimizer.param_groups[0]["lr"] = 0.00002

#     paps_eval.evaluate_paps(dataset=dataset_val, 
#                   dataloader=val_data_loader, 
#                   model=ret_model, 
#                   saved_dir=parser.saved_dir, 
#                   device = device,
#                   threshold=0.5) 
    
    for epoch in range(parser.epochs) :
        if epoch == int(parser.epochs*0.2) :
            ret_model.module.freeze_ex_bn(True)
        total_loss = 0
        tk0 = tqdm(train_data_loader, total=len(train_data_loader), leave=False)
        EPOCH_LEARING_RATE = optimizer.param_groups[0]["lr"]
        start_time = time.time()
        print("*****{}th epoch, learning rate {}".format(epoch, EPOCH_LEARING_RATE))

        for step, data in enumerate(tk0) :
            if step > len(train_data_loader)/4 and epoch < int(parser.epochs*0.8) :
                break            
            images, box, label, targets = data
            batch_size = len(images)

        #     images = list(image.to(device) for image in images)
            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)
    #         print(images.shape)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [ t.to(device) for t in targets]

    #         classification_loss, regression_loss = ret_model([images, targets])
            outputs = ret_model([images, targets])
            classification, regression, anchors, annotations = (outputs)
            classification_loss, regression_loss = criterion(classification, regression, anchors, annotations)

    #         output = ret_model(images)
    #         features, regression, classification = output
    #         classification_loss, regression_loss = criterion(classification, regression, modified_anchors, targets)    
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss 
            total_loss += loss.item()

#             if step % 100 == 0:
#                 print('lr {} batch_loss {} cls {} reg {} avg {}'.format(optimizer.param_groups[0]["lr"], loss.item(), classification_loss.item(), 
#                                 regression_loss.item(), total_loss/(step+1)
#                 ))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ret_model.parameters(), 0.02)
            optimizer.step()   

        print('{}th epochs loss is {}'.format(epoch, total_loss/(step+1)))
        if loss_per_epoch > total_loss/(step+1):
            print('best model is saved')
            torch.save(ret_model.state_dict(), parser.saved_dir + 'best_model.pt')
            loss_per_epoch = total_loss/(step+1)

        scheduler.step()
        print('epoch training time is ', time.time() - start_time)
#         if epoch % 20 == 0 :
#             paps_eval.evaluate_paps(dataset=dataset_val, 
#               dataloader=val_data_loader, 
#               model=ret_model, 
#               saved_dir=parser.saved_dir, 
#               device = device,
#               threshold=0.5)    
#         ret_model.train()

    torch.save(ret_model.state_dict(), parser.saved_dir + 'model.pt')
    paps_eval.evaluate_paps(dataset=dataset_val, 
      dataloader=val_data_loader, 
      model=ret_model, 
      saved_dir=parser.saved_dir, 
      device = device,
      threshold=0.5)  
        
if __name__ == '__main__':
    main()        