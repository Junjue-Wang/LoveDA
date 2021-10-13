import torch.nn as nn
import torch.nn.functional as F
import torch


class PPMBilinear(nn.Module):
    def __init__(self, num_classes=7, fc_dim=2048,
                 use_aux=False, pool_scales=(1, 2, 3, 6),
                 norm_layer = nn.BatchNorm2d
                 ):
        super(PPMBilinear, self).__init__()
        self.use_aux = use_aux
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)


        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )


    def forward(self, conv_out):
        #conv5 = conv_out[-1]
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)


        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            return x
        else:
            return x

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out



from module.resnet import ResNetEncoder
import ever as er
class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)
        self.encoder = ResNetEncoder(self.config.backbone)
        if self.config.multi_layer:
            print('Use multi_layer!')
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.config.multi_layer:
            if self.config.cascade:
                c3, c4 = self.encoder(x)[-2:]
                x1 = self.layer5(c3)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(c4)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x2).softmax(dim=1)
            else:

                x = self.encoder(x)[-1]
                x1 = self.layer5(x)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(x)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x1+x2).softmax(dim=1)

        else:
            feat, x = self.encoder(x)[-2:]
            #x = self.layer5(x)
            
            x = self.cls_pred(x)
            #x = self.cls_pred(x)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
            #feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
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
            ),
            multi_layer=False,
            cascade=False,
              use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,
                
            ),
            inchannels=2048,
            num_classes=7
        ))


