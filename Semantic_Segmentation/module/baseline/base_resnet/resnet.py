from functools import partial
import torch.nn as nn
from torch.utils import checkpoint as cp

from ever.core import registry, logger
from ever.interface import ERModule
from ._resnets import resnet101
from ._resnets import resnet18
from ._resnets import resnet34
from ._resnets import resnet50

from ever.util import param_util

_logger = logger.get_logger()
__all__ = ['make_layer',
           'ResNetEncoder']

registry.MODEL.register('resnet18', resnet18)
registry.MODEL.register('resnet34', resnet34)
registry.MODEL.register('resnet50', resnet50)
registry.MODEL.register('resnet101', resnet101)



def make_layer(block, in_channel, basic_out_channel, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or in_channel != basic_out_channel * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, basic_out_channel * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(basic_out_channel * block.expansion),
        )

    layers = []
    layers.append(block(in_channel, basic_out_channel, stride, dilation, downsample))
    in_channel = basic_out_channel * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channel, basic_out_channel, dilation=dilation))

    return nn.Sequential(*layers)


@registry.MODEL.register()
class ResNetEncoder(ERModule):
    def __init__(self,
                 config):
        super(ResNetEncoder, self).__init__(config)
        if all([self.config.output_stride != 16,
                self.config.output_stride != 32,
                self.config.output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')

        self.resnet = registry.MODEL[self.config.resnet_type](pretrained=self.config.pretrained,
                                                              norm_layer=self.config.norm_layer)
        _logger.info('ResNetEncoder: pretrained = {}'.format(self.config.pretrained))
        self.resnet._modules.pop('fc')
        if not self.config.batchnorm_trainable:
            self._frozen_res_bn()

        self._freeze_at(at=self.config.freeze_at)

        if self.config.output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif self.config.output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))

    def reset_in_channels(self, in_channels):
        if in_channels == 3:
            return
        self.resnet.add_module('conv1',
                               nn.Conv2d(in_channels,
                                         64, kernel_size=7, stride=2, padding=3,
                                         bias=False))

    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    def _frozen_res_bn(self):
        _logger.info('ResNetEncoder: freeze all BN layers')
        param_util.freeze_modules(self.resnet, nn.modules.batchnorm._BatchNorm)
        for m in self.resnet.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def _freeze_at(self, at=2):
        if at >= 1:
            param_util.freeze_params(self.resnet.conv1)
            param_util.freeze_params(self.resnet.bn1)
        if at >= 2:
            param_util.freeze_params(self.resnet.layer1)
        if at >= 3:
            param_util.freeze_params(self.resnet.layer2)
        if at >= 4:
            param_util.freeze_params(self.resnet.layer3)
        if at >= 5:
            param_util.freeze_params(self.resnet.layer4)

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def forward(self, inputs):
        x = inputs
        x = self.resnet.stem_forward(x)
        x = self.resnet.maxpool(x)

        # os 4, #layers/outdim: 18,34/64; 50,101,152/256
        if self.config.with_cp[0] and x.requires_grad:
            c2 = cp.checkpoint(self.get_function(self.resnet.layer1), x)
        else:
            c2 = self.resnet.layer1(x)
        # os 8, #layers/outdim: 18,34/128; 50,101,152/512
        if self.config.with_cp[1] and c2.requires_grad:
            c3 = cp.checkpoint(self.get_function(self.resnet.layer2), c2)
        else:
            c3 = self.resnet.layer2(c2)
        # os 16, #layers/outdim: 18,34/256; 50,101,152/1024
        if self.config.with_cp[2] and c3.requires_grad:
            c4 = cp.checkpoint(self.get_function(self.resnet.layer3), c3)
        else:
            c4 = self.resnet.layer3(c3)
        # os 32, #layers/outdim: 18,34/512; 50,101,152/2048
        if self.config.include_conv5:
            if self.config.with_cp[3] and c4.requires_grad:
                c5 = cp.checkpoint(self.get_function(self.resnet.layer4), c4)
            else:
                c5 = self.resnet.layer4(c4)
            return [c2, c3, c4, c5]

        return [c2, c3, c4]

    def set_default_config(self):
        self.config.update(dict(
            resnet_type='resnet50',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2d,
        ))

    def train(self, mode=True):
        super(ResNetEncoder, self).train(mode)
        self._freeze_at(self.config.freeze_at)
        if mode and not self.config.batchnorm_trainable:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def _nostride_dilate(self, m, dilate):
        # ref:
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/1235deb1d68a8f3ef87d639b95b2b8e3607eea4c/models/models.py#L256
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


