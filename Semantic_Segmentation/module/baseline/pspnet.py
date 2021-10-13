import torch
from torch import nn
from torch.nn import functional as F
import ever as er
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.loss import SegmentationLoss


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


@er.registry.MODEL.register()
class PSPNet(er.ERModule):
    def __init__(self, config):
        super().__init__(config)
        # self.feats = getattr(extractors, backend)(pretrained)
        self.feats = ResNetEncoder(self.config.encoder)

        self.psp = PSPModule(self.config.psp.psp_size, 1024, self.config.psp.sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Conv2d(64, self.config.classes, kernel_size=1)
        self.loss = SegmentationLoss(self.config.loss)

    def forward(self, x, y=None):
        h, w = x.size(2), x.size(3)
        f = self.feats(x)[-1]
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)
        logit = self.final(p)
        logit = F.upsample(input=logit, size=(h, w), mode='bilinear')
        if self.training:
            return self.loss(logit, y['cls'])

        else:
            return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=False,
                freeze_at=0,
                # 16 or 32
                output_stride=8,
                with_cp=(False, False, False, False),
                norm_layer=nn.BatchNorm2d,
            ),
            classes=7,
            loss=dict(
                ignore_index=-1,
                ce=dict()
            ),
            psp=dict(
                sizes=(1, 2, 3, 6),
                psp_size=2048,
                deep_features_size=1024
            )
        ))

if __name__ == '__main__':
    m = PSPNet(dict())
    m.eval()
    o = m(torch.ones(2, 3, 512, 512))
    print(o.shape)