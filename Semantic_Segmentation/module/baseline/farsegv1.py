import ever as er
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.baseline.base import FPN, AssymetricDecoder, FSRelation
from segmentation_models_pytorch.encoders import get_encoder



from module.loss import SegmentationLoss
@er.registry.MODEL.register()
class FarSegV1(er.ERModule):
    # FarSegV1: ResNet + FPN + FSR + (normalized focal loss)
    # FarSegV1plus: ResNet v1d + PPM + FPN + Improved FSR + (normalized focal loss)
    def __init__(self, config):
        super(FarSegV1, self).__init__(config)
        self.en = get_encoder(self.config.backbone.type, weights=self.config.backbone.weights)
        self.fsr = FSRelation(**self.config.fs_relation)
        self.fpn = FPN(**self.config.fpn)
        self.seg_decoder = AssymetricDecoder(**self.config.asy_decoder)
        self.loss = SegmentationLoss(self.config.loss)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4.0)
        self.cls_pred_conv = nn.Conv2d(self.config.asy_decoder.out_channels, self.config.classes, 1)

    def forward(self, x, y=None):
        feature_list = self.en(x)
        last_feat = feature_list[-1]
        # fpn
        fpn_feature_list = self.fpn(feature_list)
        # fsr
        scene_embedding = F.adaptive_avg_pool2d(last_feat, 1)
        refined_fpn_feature_list = self.fsr(scene_embedding, fpn_feature_list)
        # decode
        logit = self.seg_decoder(refined_fpn_feature_list)
        logit = self.cls_pred_conv(logit)
        logit = self.upsample4x_op(logit)
        if self.training:
            return self.loss(logit, y['cls'])

        else:
            return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                type='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            ppm=dict(
                in_channels=2048,
                pool_channels=512,
                out_channels=512,
                bins=(1, 2, 3, 6),
                bottleneck_conv='1x1',
                dropout=0.1
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 512),
                out_channels=256,
            ),
            fs_relation=dict(
                scene_embedding_channels=2048,
                in_channels_list=(256, 256, 256, 256),
                out_channels=256,
                scale_aware_proj=True
            ),
            asy_decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                
            ),
            loss=dict(
            ),
        ))

    def log_info(self):
        return dict(cfg=self.config)
