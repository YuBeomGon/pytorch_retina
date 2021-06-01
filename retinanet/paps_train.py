import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from retinanet.scheduler import *
from retinanet.losses import *
from retinanet.dataloader import *
from retinanet.anchors import Anchors

import albumentations as A
import albumentations.pytorch
import time
from tqdm import tqdm

def train_paps(dataloader, model, saved_dir, criterion,
                  optimizer, scheduler, device, s_epoch=0, e_epoch=160, last_loss=0.6):
    
    #for i, data in enumerate(tqdm(train_data_loader)) :
    train_data_loader = dataloader
    if s_epoch == 0:
        Iou_low = [0.2]
        beta_list = [1.0]
        for i in range(1, e_epoch+1) :
            Iou_low.append(Iou_low[0] + (0.5-0.2)*i/160)
            beta_list.append(beta_list[0] - (1.0-0.5)*i/160)   
            
    loss_per_epoch = last_loss
    cls_loss = []
    box_loss = []
    for epoch in range(s_epoch, e_epoch) :
        total_loss = 0
        tk0 = tqdm(train_data_loader, total=len(train_data_loader), leave=False)
        EPOCH_LEARING_RATE = optimizer.param_groups[0]["lr"]

        for step, data in enumerate(tk0) :
            if step > len(train_data_loader)/2 and epoch%10 > 0 :
#             if step > len(train_data_loader)/4 :
                break 
            if s_epoch == 0 :
                iou_thres = np.random.uniform(low=Iou_low[epoch], high=0.5, size=None)
                beta = np.random.uniform(low=beta_list[epoch]-0.1, high=beta_list[epoch], size=None)
            else :
                iou_thres = 0.5
                beta = 0.5
            images, box, label, targets = data
            batch_size = len(images)

            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)
            targets = [ t.to(device) for t in targets]

            outputs = model([images, targets])
            classification, regression, anchors, annotations = (outputs)
            classification_loss, regression_loss, num_det = criterion(classification, regression, 
                                                             anchors, annotations, iou_thres,
                                                             beta)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss 
            total_loss += loss.item()
            cls_loss.append(classification_loss.item())
            box_loss.append(regression_loss.item())

            if step % 5 == 0:
                tk0.set_postfix(lr=optimizer.param_groups[0]["lr"], batch_loss=loss.item(), cls_loss=classification_loss.item(), 
                                reg_loss=regression_loss.item(), avg_loss=total_loss/(step+1), num_det=num_det)        

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.02)
            optimizer.step()   

        print('{}th epochs loss {} lr {} '.format(epoch, total_loss/(step+1), EPOCH_LEARING_RATE ))
        if loss_per_epoch > total_loss/(step+1):
            print('best model is saved')
            torch.save(model.state_dict(), saved_dir + 'best_model.pt')
            loss_per_epoch = total_loss/(step+1)

        scheduler.step()


    #     print('epoch training time is ', time.time() - start_time)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss' : total_loss/(step+1),
        'scheduler' : scheduler.state_dict()
    }
    if os.path.isdir(saved_dir) == False :
        print('saved_dir is made')
        os.makedirs(saved_dir)    
    torch.save(state, saved_dir + 'epoch_' + str(e_epoch) +'_model.pt')