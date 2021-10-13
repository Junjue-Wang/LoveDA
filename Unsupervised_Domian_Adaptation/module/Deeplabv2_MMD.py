import ever as er
from module.resnet import ResNetEncoder
import torch.nn as nn
import torch.nn.functional as F
from module.Encoder import Classifier_Module


class ReductionLayer(nn.Module):
    def __init__(self, inplanes):
        super(ReductionLayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(inplanes, inplanes // 4)      # 2048 - > 512
        self.l2 = nn.Linear(inplanes // 4, inplanes // 8) # 512 - > 256
        self.relu = nn.ReLU()

    def forward(self, feat):
        feat = self.gap(feat)
        flatten_feat = feat.view(feat.size(0), -1)
        reduction_feat = self.l1(flatten_feat)
        reduction_feat = self.relu(reduction_feat)
        reduction_feat = self.l2(reduction_feat)
        reduction_feat = self.relu(reduction_feat)
        return reduction_feat


class Deeplabv2_MMD(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2_MMD, self).__init__(config)
        self.encoder = ResNetEncoder(self.config.backbone)
        # self.cls_pred = nn.Conv2d(self.config.inchannels, self.config.num_classes, 1, 1)
        self.cls_pred = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
        self.reduction_layer = ReductionLayer(self.config.inchannels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.encoder(x)[-1]
        feat = self.reduction_layer(x)
        x = self.cls_pred(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        if self.training:
            return x, feat
        else:
            return x.softmax(dim=1)


    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
                multi_layer = False,
            ),
            inchannels=2048,
            num_classes=7
        ))
