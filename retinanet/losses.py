import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    
#     area_b = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def paps_calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua
    IoU_paps = intersection / area
    IoU_paps = torch.clamp(IoU_paps, min=0, max=1)
    
    return (IoU + IoU_paps) / 2

class FocalLoss(nn.Module):
    def __init__(self,device):
        super(FocalLoss, self).__init__()
        self.device = device
        self.alpha = 0.25
        self.gamma = 2.0   

    def forward(self, classifications, regressions, anchors, annotations, iou_threshold=0.3, beta=1.0, pimages=None):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        num_detected = 0

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

#             bbox_annotation = annotations[j, :, :]
            bbox_annotation = annotations[j]
            
            if len(bbox_annotation) == 0 :
                pass
#                 print('No Bounding Box found {}'.format(bbox_annotation))
            else :
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).to(self.device) * self.alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, self.gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().to(self.device))

                else:
                    alpha_factor = torch.ones(classification.shape) * self.alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, self.gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for clIoU_max.shape, assification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.to(self.device)

            targets[torch.lt(IoU_max, iou_threshold), :] = 0

            positive_indices = torch.ge(IoU_max, iou_threshold + 0.1)

            num_positive_anchors = positive_indices.sum()
            num_detected += num_positive_anchors

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).to(self.device) * self.alpha
            else:
                alpha_factor = torch.ones(targets.shape) * self.alpha
                
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, self.gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(self.device))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.device)
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

#                 change the beta value from 1/9 to 1.0
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0),
                    0.5 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().to(self.device))
                else:
                    regression_losses.append(torch.tensor(0).float())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True), num_detected

class PapsLoss(FocalLoss) :
    def __init__(self,device, target_threshold=0.9, topk=5, filter_option=4):
        super(PapsLoss, self).__init__(device)
#         self.device = device    
        self.cell_threshold = 0.5
        self.target_threshold = target_threshold
        self.topk = topk
        self.filter_option = filter_option   
        self.pimage_thres = 0.55

    def forward(self, classifications, regressions, anchors, annotations, iou_threshold=0.5, beta=1.0, pimages=None):

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
#         print('anchors', anchors.shape, anchors)
#         print('annotations', annotations[0])

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        num_detected = 0
        avg_foreground = []
        avg_background = []

        for j in range(batch_size):

            classification = classifications[j, :, :]
            pimage = pimages[j, :]
            regression = regressions[j, :, :]

#             bbox_annotation = annotations[j, :, :]
            bbox_annotation = annotations[j]
            
            if len(bbox_annotation) == 0 :
                pass
#                 print('No Bounding Box found {}'.format(bbox_annotation))
            else :
                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).to(self.device) * self.alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, self.gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().to(self.device))

                else:
                    alpha_factor = torch.ones(classification.shape) * self.alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, self.gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

#             IoU = paps_calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

            #import pdb
            #pdb.set_trace()

            # compute the loss for clIoU_max.shape, assification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.to(self.device)

            targets[torch.lt(IoU_max, iou_threshold), :] = 0
            
#             loss filtering(1~3), label smoothing 5 and image filtering 6
            if int(self.filter_option) == 1 :
                targets[classification[:,1] > self.target_threshold, 1] = -1
            elif int(self.filter_option) == 2 :
                topk_value = torch.topk(classification[:,1], self.topk)[0][self.topk-1]
                targets[classification[:,1] > topk_value, 1] = -1
            elif int(self.filter_option) == 3 :
                topk_value = torch.topk(classification[:,1], self.topk)[0][self.topk-1]
                if topk_value < self.target_threshold :
                    topk_value = self.target_threshold
                targets[classification[:,1] > topk_value, 1] = -1    
            elif int(self.filter_option) == 5 :
                targets[classification[:,1] > self.target_threshold, 1] = classification[classification[:,1] > self.target_threshold, 1] - self.target_threshold
                targets[classification[:,0] > (self.target_threshold+0.1), 0] = classification[classification[:,0] > (self.target_threshold+0.1), 0] - (self.target_threshold+0.1)  
            elif int(self.filter_option) == 6 and pimage != None :
#                 pimage[torch.lt(IoU_max, iou_threshold)]
                if len(pimage[torch.lt(IoU_max, 0.1)]) > 0 :
                    avg_background.extend(pimage[torch.lt(IoU_max, 0.1)])
                if len(pimage[torch.ge(IoU_max, 0.9)]) > 0 :
                    avg_foreground.extend(pimage[torch.ge(IoU_max, 0.9)])
#                 print(pimage[torch.ge(IoU_max, 0.5)])
                targets[pimage[:] < self.pimage_thres, :] = -1

            positive_indices = torch.ge(IoU_max, iou_threshold + 0.1)

            num_positive_anchors = positive_indices.sum()
            num_detected += num_positive_anchors

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            c_weight = targets * (IoU_max.unsqueeze(dim=0).transpose(1,0)+0.5)
            c_weight = torch.clamp(c_weight, min=0.5)

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).to(self.device) * self.alpha
            else:
                alpha_factor = torch.ones(targets.shape) * self.alpha
            
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
#             Use C weight based on IoU instead of alpha factor which is fixed
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, self.gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ge(targets, -0.1), cls_loss, torch.zeros(cls_loss.shape).to(self.device))
            else:
                cls_loss = torch.where(torch.ge(targets, -0.1), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
#                 targets_dw = (gt_widths / anchor_widths_pi)
#                 targets_dh = (gt_heights / anchor_heights_pi)                

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

#                 if torch.cuda.is_available():
#                     targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(self.device)
#                 else:
#                     targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

#                 change the beta value from 1/9 to 1.0
                regression_loss = torch.where(
                    torch.le(regression_diff, beta),
                    0.5 * torch.pow(regression_diff, 2)/beta,
                    regression_diff - 0.5*beta
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().to(self.device))
                else:
                    regression_losses.append(torch.tensor(0).float())
#         if len(avg_foreground) == 0:
#             avg_fore= torch.zeros(1).to(self.device)
#         else :
#             avg_fore = sum(avg_foreground)/len(avg_foreground)
#         if len(avg_background) == 0:
#             avg_back = torch.zeros(1).to(self.device)
#         else :
#             avg_back = sum(avg_background)/len(avg_background)
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True), num_detected, avg_background, avg_foreground