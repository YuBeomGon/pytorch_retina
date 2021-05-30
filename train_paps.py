import argparse
import collections

import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageEnhance
# import albumentations as A
# import albumentations.pytorch
# from tqdm.notebook import tqdm
from tqdm import tqdm

import sys
sys.path.append('../')
from retinanet import model
from retinanet import coco_eval
from retinanet import paps_eval
from retinanet import csv_eval
from retinanet import paps_train

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
# from apex.parallel import DistributedDataParallel as DDP

from pycocotools.cocoeval import COCOeval
import json
import torch
import torchvision.models as models

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple paps training script for training a RetinaNet network.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)
    parser.add_argument('--start_epoch', help='start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', help='end_epoch', type=int, default=200)
    parser.add_argument('--batch_size', help='Number of batchs', type=int, default=64)    
    parser.add_argument('--train_data', help='train data file', default='data/train.npy')    
    parser.add_argument('--test_data', help='test data file', default='data/test.npy')   
    parser.add_argument('--saved_dir', help='saved dir', default='trained_models/resnet101_320/') 
    parser.add_argument('--gpu_num', help='default gpu', type=int, default=3) 
    parser.add_argument('--ismultigpu', help='multi gpu support', type=bool, default=False) 
    parser.add_argument('--freeze_ex_bn', help='freeze batch norm', type=bool, default=False) 
    parser.add_argument('--num_workers', help='cpu core', type=int, default=12) 
    
    parser = parser.parse_args(args)
    print('batch_size ', parser.batch_size)
    print('epochs ', parser.epochs)
    print( ' start_epoch {} end_epoch {}'.format(parser.start_epoch, parser.end_epoch))
    print('ismultigpu', parser.ismultigpu)
    print('freeze_ex_bn', parser.freeze_ex_bn )

    # GPU 할당 변경하기
    GPU_NUM = parser.gpu_num
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    
    resnet101 = models.resnet101(progress=False, pretrained=True)    
    ret_model = model.resnet101(num_classes=2, device=device)
    ret_model.load_state_dict(resnet101.state_dict(), strict=False)   
    
#     In Batch norm initial setting, set r to and set to 1 for bias for fast convergence
    state_dict = ret_model.state_dict()
    for s in state_dict:
        if 'bn' in s and 'residualafterFPN' in s :
            if 'weight' in s :
                shape = state_dict[s].shape
                state_dict[s] = torch.zeros(shape)
            elif 'bias' in s :
                shape = state_dict[s].shape
                state_dict[s] = torch.ones(shape)            
    
    ret_model.load_state_dict(state_dict)
    
#     criterion = FocalLoss(device)
    criterion = PapsLoss(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(ret_model.parameters(), lr = 1e-7)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=1, eta_max=0.0004,  T_up=5, gamma=0.5)    
    
    saved_dir = parser.saved_dir
    if os.path.isfile(saved_dir+'model.pt') :
        state = torch.load(saved_dir + 'model.pt')
        ret_model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        last_loss = state['loss'] 
        scheduler.load_state_dict(state['scheduler'])
    else :
        last_loss = 0.6 
    
    if parser.ismultigpu :
        ret_model = torch.nn.DataParallel(ret_model, device_ids = [3,4,5], output_device=GPU_NUM).to(device)
    # ret_model = DataParallelModel(ret_model, device_ids = device_ids)
    ret_model.to(device)
#     ret_model.module.freeze_bn()     
    
    batch_size = parser.batch_size
    dataset_train = PapsDataset('data/', set_name='train_2class',
                                transform=train_transforms)

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=parser.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    dataset_val = PapsDataset('data/', set_name='val_2class',
                                transform=val_transforms)

    val_data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )     
    
    s_epoch= parser.start_epoch
    e_epoch= parser.end_epoch
    ret_model.training = True
    
    paps_train.train_paps(dataloader=train_data_loader, 
                          model=ret_model, 
                          criterion=criterion,
                          saved_dir=saved_dir, 
                          optimizer=optimizer,
                          scheduler=scheduler,
                          device = device,
                          s_epoch= s_epoch,
                          e_epoch= e_epoch,
                          last_loss = last_loss) 
    
    ret_model.training = False
#     ret_model.eval()

    paps_eval.evaluate_paps(dataset=dataset_val, 
      dataloader=val_data_loader, 
      model=ret_model, 
      saved_dir=parser.saved_dir, 
      device = device,
      threshold=0.5)  
        
if __name__ == '__main__':
    print(sys.argv)
    main()        