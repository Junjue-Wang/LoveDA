import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ever.interface import ERModule
from ever import registry
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.baseline.base import AssymetricDecoder, FPN, default_conv_block
import math
from module.loss import SegmentationLoss, multi_binary_loss



@registry.MODEL.register('SemanticFPN')
class SemanticFPN(ERModule):
    def __init__(self, config):
        super(SemanticFPN, self).__init__(config)
        self.en = ResNetEncoder(self.config.resnet_encoder)
        self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)


    def forward(self, x, y=None):
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        final_feat = self.decoder(fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)
        if self.training:
            cls_true = y['cls']
            #loss_dict = dict()
            loss_dict = self.cls_loss(cls_pred, cls_true)
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict

        cls_prob = torch.softmax(cls_pred, dim=1)

        return cls_prob



    def set_default_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                norm_layer=nn.BatchNorm2d,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 256),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=7,
            loss=dict(
                ignore_index=255,
            )
        ))




@registry.MODEL.register()
class SemanticFPNDecouple(ERModule):
    def __init__(self, config):
        super(SemanticFPNDecouple, self).__init__(config)
        self.en = ResNetEncoder(self.config.resnet_encoder)
        self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes-1, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)


    def forward(self, x, y=None):
        feat_list = self.en(x)
        fpn_feat_list = self.fpn(feat_list)
        final_feat = self.decoder(fpn_feat_list)
        pred = self.cls_pred_conv(final_feat)
        pred = self.upsample4x_op(pred)
        if self.training:
            y_true = y['cls'] 
            return multi_binary_loss(pred, y_true, self.config.classes -1, dice_scaler=self.config.loss.bce_scaler, bce_scaler=self.config.loss.dice_scaler, label_smooth=self.config.loss.label_smooth)
        else:
            return pred.sigmoid()




    def set_default_config(self):
        self.config.update(dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                norm_layer=nn.BatchNorm2d,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 256),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=7,
            loss=dict(
                bce_scaler=1.0,
                dice_scaler=1.0,
                label_smooth=0.
            )
        ))

