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

import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP

from pycocotools.cocoeval import COCOeval
import json
import torch

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple paps training script for training a RetinaNet network.')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=160)
    parser.add_argument('--batch_size', help='Number of batchs', type=int, default=128)    
    parser.add_argument('--train_data', help='train data file', default='data/train.npy')    
    parser.add_argument('--test_data', help='test data file', default='data/test.npy')        
    
    parser = parser.parse_args(args)

    # GPU 할당 변경하기
    GPU_NUM = 0 # 원하는 GPU 번호 입력
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print(device)
    print ('Current cuda device ', torch.cuda.current_device()) # check
    device_ids = [0,1,2,4]
    
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
        
    retinanet = torch.nn.DataParallel(retinanet, device_ids = [0,1,2,4], output_device=0).to(device)
    # retinanet = DataParallelModel(retinanet, device_ids = device_ids)
    retinanet.to(device)
    # retinanet.cuda()
#     retinanet.module.freeze_bn()     
    
    batch_size = parser.batch_size
    dataset_train = PapsDataset('data/', set_name='train_2class',
                                transform=train_transforms)

    train_data_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=collate_fn
    )

    criterion = FocalLoss(device)
    criterion = criterion.to(device)
    retinanet.training = True
    
    optimizer = optim.Adam(retinanet.parameters(), lr = 1e-7)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_max=0.0004,  T_up=7, gamma=0.5)    
    
    loss_per_epoch = 0.6
    for epoch in range(parser.epochs) :
        total_loss = 0
        tk0 = tqdm(train_data_loader, total=len(train_data_loader), leave=False)
        EPOCH_LEARING_RATE = optimizer.param_groups[0]["lr"]
        start_time = time.time()
        print("*****{}th epoch, learning rate {}".format(epoch, EPOCH_LEARING_RATE))

        for step, data in enumerate(tk0) :
            images, box, label, targets = data
            batch_size = len(images)

        #     images = list(image.to(device) for image in images)
            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)
    #         print(images.shape)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [ t.to(device) for t in targets]

    #         classification_loss, regression_loss = retinanet([images, targets])
            outputs = retinanet([images, targets])
            classification, regression, anchors, annotations = (outputs)
            classification_loss, regression_loss = criterion(classification, regression, anchors, annotations)

    #         output = retinanet(images)
    #         features, regression, classification = output
    #         classification_loss, regression_loss = criterion(classification, regression, modified_anchors, targets)    
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss 
            total_loss += loss.item()

            if step % 50 == 0:
                tk0.set_postfix(lr=optimizer.param_groups[0]["lr"], batch_loss=loss.item(), cls_loss=classification_loss.item(), 
                                reg_loss=regression_loss.item(), avg_loss=total_loss/(step+1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()   

        print('{}th epochs loss is {}'.format(epoch, total_loss/(step+1)))
        if loss_per_epoch > total_loss/(step+1):
            print('best model is saved')
            torch.save(retinanet.state_dict(), 'trained_models/best_model.pt')
            loss_per_epoch = total_loss/(step+1)

        scheduler.step()
        print('epoch training time is ', time.time() - start_time)

    torch.save(retinanet.state_dict(), 'trained_models/model.pt')
    
    dataset_val = PapsDataset('data/', set_name='val_2class',
                                transform=val_transforms)

    val_data_loader = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )    
    
    retinanet.eval()
    start_time = time.time()
    threshold = 0.1
    results = []
    GT_results = []
    image_ids = []
    cnt = 0
    scores_list = []

    for index, data in enumerate(tqdm(val_data_loader)) :
    #     if cnt > 500 :
    #         break
    #     cnt += 1
        with torch.no_grad():        
            images, tbox, tlabel, targets = data
            batch_size = len(images)
    #         print(tbox)
    #         print(len(tbox[0]))

            c, h, w = images[0].shape
            images = torch.cat(images).view(-1, c, h, w).to(device)

            outputs = retinanet(images)
            scores, labels, boxes = (outputs)

            scores = scores.cpu()
            labels = labels.cpu()
            boxes  = boxes.cpu()  

            scores_list.append(scores)

            if boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
    #             print(boxes)

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset_val.image_ids[index],
                        'category_id' : dataset_val.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            if len(tbox[0]) > 0:    

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(len(tbox[0])):
                    score = float(0.99)
                    label = (tlabel[0][box_id])
                    box = list(tbox[0][box_id])
                    box[2] -= box[0]
                    box[3] -= box[1]             

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset_val.image_ids[index],
                        'category_id' : dataset_val.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : list(box),
                    }

                    # append detection to results
                    GT_results.append(image_result)                

            # append image to list of processed images
            image_ids.append(dataset_val.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset_val)), end='\r')    

    if not len(results):
        print('No object detected')
    print('GT_results', len(GT_results))    
    print('pred_results', len(results))    

    # write output
    json.dump(results, open('trained_models/eval/{}_bbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)
    # write GT
    json.dump(GT_results, open('trained_models/eval/{}_GTbbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)     

    print('validation time :', time.time() - start_time)


    # load results in COCO evaluation tool
    coco_true = dataset_val.coco
    coco_pred = coco_true.loadRes('trained_models/eval/{}_bbox_results.json'.format(dataset_val.set_name))
    coco_gt = coco_true.loadRes('trained_models/eval/{}_GTbbox_results.json'.format(dataset_val.set_name))

    # run COCO evaluation
    # coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
    print("******************total*********************")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    # coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()    
    
    print("******************Abnormal*********************")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [0]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()   
    
    print("******************Normal*********************")
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()       
        
if __name__ == '__main__':
    main()        