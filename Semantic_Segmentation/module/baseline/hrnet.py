from module.baseline.base_hrnet.hrnet_encoder import HRNetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from ever.core import registry
import ever as er
from module.loss import SegmentationLoss
BatchNorm2d = nn.BatchNorm2d

BN_MOMENTUM = 0.1


class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, feat_list):
        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(feat_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(feat_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(feat_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        return x

@registry.MODEL.register('HRNetFusion')
class HRNetFusion(er.ERModule):
    def __init__(self, config):
        super(HRNetFusion, self).__init__(config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = HRNetEncoder(self.config.backbone)
        self.neck = SimpleFusion(self.config.neck.in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale),
        )
        self.loss = SegmentationLoss(self.config.loss)

    def forward(self, x, y=None):
        pred_list = self.backbone(x)

        logit = self.neck(pred_list)
        logit = self.head(logit)
        if self.training:
            y_true = y['cls']
            return self.loss(logit, y_true.long())
        else:
            return logit.softmax(dim=1)


    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                hrnet_type='hrnetv2_w48',
                pretrained=False,
                norm_eval=False,
                frozen_stages=-1,
                with_cp=False,
                with_gc=False,
            ),
            neck=dict(
                in_channels=720,
            ),
            classes=7,
            head=dict(
                in_channels=720,
                upsample_scale=4.0,
            ),
            loss=dict(
                
                ce=dict(),
            )
        ))


