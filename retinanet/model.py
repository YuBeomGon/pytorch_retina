import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.utils import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def predict (anchors, regression, classification, device, regressBoxes, clipBoxes, img_batch) :
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, img_batch)

    finalResult = [[], [], []]

    finalScores = torch.Tensor([])
    finalAnchorBoxesIndexes = torch.Tensor([]).long()
    finalAnchorBoxesCoordinates = torch.Tensor([])

    if torch.cuda.is_available():
        finalScores = finalScores.to(device)
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.to(device)
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.to(device)


    for i in range(classification.shape[2]):
        scores = torch.squeeze(classification[:, :, i])
        scores_over_thresh = (scores > 0.05)
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just continue
            continue

        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(transformed_anchors)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

        finalResult[0].extend(scores[anchors_nms_idx])
        finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
        finalResult[2].extend(anchorBoxes[anchors_nms_idx])

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
        if torch.cuda.is_available():
            finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.to(device)

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]    

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, device):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.device = device

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.residualafterFPN = ResidualAfterFPN()     
        
        self.anchors = Anchors()
        print('num_anchors per feature map', self.anchors.num_anchors)

        self.regressionModel = RegressionModel(256, num_anchors=self.anchors.num_anchors)
        self.classificationModel = ClassificationModel(256, num_anchors=self.anchors.num_anchors, num_classes=num_classes)

        self.regressBoxes = BBoxTransform(device=self.device)

        self.clipBoxes = ClipBoxes()

#         self.focalLoss = losses.FocalLoss(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

#         self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
            
    def freeze_ex_bn(self, req_gra) :
        for n, p in self.named_parameters():
            #print(n)
            if 'bn' not in n :
                if 'fpn' not in n and 'residualafterFPN' not in n and 'regressionModel' not in n and 'classificationModel' not in n :
                    #print(param)
                    #print(n)
                    p.requires_grad = req_gra

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        x = self.bn0(img_batch)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        
        features = [self.residualafterFPN(feature) for feature in features] 

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)
        anchors = anchors.to(self.device)

        if self.training:
#             return self.focalLoss(classification, regression, anchors, annotations)
            return classification, regression, anchors, annotations
        else:
            return predict(anchors, regression, classification, 
                           self.device, self.regressBoxes, self.clipBoxes, img_batch)

def resnet18(num_classes, device, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, device, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, device, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, device, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, device, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], device=device, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model
