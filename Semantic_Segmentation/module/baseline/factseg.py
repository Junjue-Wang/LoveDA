import ever as er
from module.baseline.semantic_fpn import FPN, AssymetricDecoder
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder
from module.loss import binary_cross_entropy_with_logits
import torch

@er.registry.MODEL.register()
class FactSeg(er.ERModule):
    def __init__(self, config):
        super(FactSeg, self).__init__(config)
        self.en = get_encoder(self.config.backbone.type, weights=self.config.backbone.weights)
        # encoder attention
        print('use fpn!')
        self.fgfpn = FPN(**self.config.foreground.fpn)
        self.bifpn = FPN(**self.config.binary.fpn)
        # decoder
        self.fg_decoder = AssymetricDecoder(**self.config.foreground.assymetric_decoder)
        self.bi_decoder = AssymetricDecoder(**self.config.binary.assymetric_decoder)

        self.fg_cls = nn.Conv2d(self.config.foreground.out_channels, self.config.classes, kernel_size=1)
        self.bi_cls = nn.Conv2d(self.config.binary.out_channels, 1, kernel_size=1)


    def forward(self, x, y=None):
        feat_list = self.en(x)[1:]
        if 'skip_decoder' in self.config.foreground:
            fg_out = self.fgskip_deocder(feat_list)
            bi_out = self.bgskip_deocder(feat_list)
        else:
            forefeat_list = list(self.fgfpn(feat_list))
            binaryfeat_list = self.bifpn(feat_list)

            fg_out = self.fg_decoder(forefeat_list)
            bi_out = self.bi_decoder(binaryfeat_list)

        fg_pred = self.fg_cls(fg_out)
        bi_pred = self.bi_cls(bi_out)
        fg_pred = F.interpolate(fg_pred, scale_factor=4.0, mode='bilinear',
                                align_corners=True)
        bi_pred = F.interpolate(bi_pred, scale_factor=4.0, mode='bilinear',
                                align_corners=True)
        if self.training:
            cls_true = y['cls']
            cls_loss = F.cross_entropy(fg_pred, cls_true.long(), ignore_index=-1)
            bi_true = torch.where(cls_true>0, torch.ones_like(cls_true), torch.zeros_like(cls_true))
            bi_true[cls_true == -1] = -1
            bi_loss = binary_cross_entropy_with_logits(bi_pred, bi_true, ignore_index=-1)
            return dict(cls_loss=cls_loss, bi_loss=bi_loss)
        else:
            return fg_pred.softmax(dim=1)


    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                type='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            classes=7,
            foreground=dict(
                fpn=dict(
                    in_channels_list=[256, 512, 1024, 2048],
                    out_channels=256,
                ),
                assymetric_decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                ),
                out_channels=128,
            ),
            binary = dict(
                fpn=dict(
                    in_channels_list=[256, 512, 1024, 2048],
                    out_channels=256,
                ),
                out_channels=128,
                assymetric_decoder=dict(
                    in_channels=256,
                    out_channels=128,
                    in_feat_output_strides=(4, 8, 16, 32),
                    out_feat_output_stride=4,
                ),
            ),
        ))