import ever as er
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.baseline.base import FPN, AssymetricDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.encoders import get_encoder


class JointLoss(nn.Module):
    def __init__(self, ignore_index=-1, sample=None, ratio=0.):
        super(JointLoss, self).__init__()
        self.ignore_index = ignore_index
        self.sample = sample
        self.ratio = ratio
        print('Sample:', sample)
       

    def forward(self, cls_pred, binary_pred, cls_true, instance_mask=None):
        valid_mask = (cls_true != self.ignore_index)
        fgp = torch.sigmoid(binary_pred)
        clsp = torch.softmax(cls_pred, dim=1)
        # numerator
        joint_prob = torch.clone(clsp)
        joint_prob[:, 0, :, :] = (1-fgp).squeeze(dim=1) * clsp[:, 0, :, :]
        joint_prob[:, 1:, :, :] = fgp * clsp[:, 1:, :, :]
        # # normalization factor, [B x 1 X H X W]
        Z = torch.sum(joint_prob, dim=1, keepdim=True)
        # cls prob, [B, N, H, W]
        p_ci = joint_prob / (Z + 1e-8)

        losses = F.nll_loss(torch.log(p_ci), cls_true.long(), ignore_index=self.ignore_index, reduction='none')
        
        if self.sample == 'SOM':
            return som(losses, self.ratio)
        elif self.sample == 'OHEM':
            seg_weight = ohem_weight(p_ci, cls_true.long(), thresh=self.ratio)
            return (seg_weight * losses).sum() / seg_weight.sum()
        else:
            return losses.sum() / valid_mask.sum()

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

        # loss
        if 'joint_loss' in self.config.loss:
            self.joint_loss = JointLoss(**self.config.loss.joint_loss)
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
            if 'joint_loss' in self.config.loss:
                return dict(joint_loss =self.joint_loss(fg_pred, bi_pred, cls_true))
            else:
                return self.cls_loss(fg_pred, bi_pred.squeeze(dim=1), cls_true)
        
        else:
            if 'joint_loss' in self.config.loss:
                binary_prob = torch.sigmoid(bi_pred)
                cls_prob = torch.softmax(fg_pred, dim=1)
                cls_prob[:, 0, :, :] = cls_prob[:, 0, :, :] * (1- binary_prob).squeeze(dim=1)
                cls_prob[:, 1:, :, :] = cls_prob[:, 1:, :, :] * binary_prob
                Z = torch.sum(cls_prob, dim=1, keepdim=True)
                cls_prob = cls_prob.div_((Z+ 1e-8))
                return cls_prob
            else:
                return torch.softmax(fg_pred, dim=1)



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
            loss=dict(
              joint_loss=dict(
                
              )
            )
        ))