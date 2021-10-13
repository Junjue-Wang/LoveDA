import segmentation_models_pytorch as smp
import ever as er
from module.loss import SegmentationLoss


@er.registry.MODEL.register()
class AnyUNet(er.ERModule, ):
    def __init__(self, config):
        super(AnyUNet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.Unet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)
        
        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))



@er.registry.MODEL.register()
class UNetPP(er.ERModule, ):
    def __init__(self, config):
        super(UNetPP, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.UnetPlusPlus(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))


@er.registry.MODEL.register()
class LinkNet(er.ERModule, ):
    def __init__(self, config):
        super(LinkNet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.Linknet(self.config.encoder_name,
                                 encoder_weights=self.config.encoder_weights,
                                 classes=self.config.classes,
                                 activation=None
                                 )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))



@er.registry.MODEL.register()
class DeepLabV3(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.DeepLabV3(self.config.encoder_name,
                                    encoder_weights=self.config.encoder_weights,
                                    classes=self.config.classes,
                                    activation=None
                                    )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))


@er.registry.MODEL.register()
class DeepLabV3Plus(er.ERModule, ):
    def __init__(self, config):
        super(DeepLabV3Plus, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.DeepLabV3Plus(self.config.encoder_name,
                                      encoder_weights=self.config.encoder_weights,
                                      classes=self.config.classes,
                                      activation=None
                                      )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))




@er.registry.MODEL.register()
class MANet(er.ERModule, ):
    def __init__(self, config):
        super(MANet, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.MAnet(self.config.encoder_name,
                                   encoder_weights=self.config.encoder_weights,
                                   classes=self.config.classes,
                                   activation=None
                                   )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))

@er.registry.MODEL.register()
class PAN(er.ERModule, ):
    def __init__(self, config):
        super(PAN, self).__init__(config)
        self.loss = SegmentationLoss(self.config.loss)
        self.features = smp.PAN(self.config.encoder_name,
                                  encoder_weights=self.config.encoder_weights,
                                  classes=self.config.classes,
                                  activation=None
                                  )

    def forward(self, x, y=None):
        logit = self.features(x)

        if self.training:
            return self.loss(logit, y['cls'])

        return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            encoder_name='resnet50',
            classes=1,
            encoder_weights=None,
            loss=dict(
                ce=dict()
            )
        ))

