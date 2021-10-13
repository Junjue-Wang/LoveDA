import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16
from ever.interface import ERModule
from ever import registry
import torch
from module.loss import SegmentationLoss


@registry.MODEL.register()
class FCN8s(ERModule):
    def __init__(self, config):
        super().__init__(config)
        # self.aux = aux
        self.loss = SegmentationLoss(self.config.loss)
        self.pretrained = vgg16(pretrained=self.config.pretrained).features

        self.pool3 = nn.Sequential(*self.pretrained[:17])
        self.pool4 = nn.Sequential(*self.pretrained[17:24])
        self.pool5 = nn.Sequential(*self.pretrained[24:])
        self.head = _FCNHead(512, self.config.classes, nn.BatchNorm2d)
        self.score_pool3 = nn.Conv2d(256, self.config.classes, 1)
        self.score_pool4 = nn.Conv2d(512, self.config.classes, 1)
        # if aux:
        #     self.auxlayer = _FCNHead(512, nclass, norm_layer)


    def forward(self, x, y=None):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        # outputs = []
        score_fr = self.head(pool5)

        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore2 = F.interpolate(score_fr, score_pool4.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool4 = upscore2 + score_pool4

        upscore_pool4 = F.interpolate(fuse_pool4, score_pool3.size()[2:], mode='bilinear', align_corners=True)
        fuse_pool3 = upscore_pool4 + score_pool3

        cls_pred = F.interpolate(fuse_pool3, x.size()[2:], mode='bilinear', align_corners=True)


        if self.training:
            return self.loss(cls_pred, y['cls'])

        return cls_pred.softmax(dim=1)



    def set_default_config(self):
        self.config.update(dict(
            classes=7,
            pretrained=True,
            loss=dict(
                ignore_index=-1
            ),
        ))

class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)



if __name__ == '__main__':
    model = FCN8s(dict())
    x = torch.ones(2, 3, 512, 512)
    y = torch.ones(2, 512, 512)
    print(model(x, dict(cls=y.long())))