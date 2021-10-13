import torch
import torch.nn as nn
import ever as er

__all__ = [
    'tta',
    'TestTimeAugmentation',
    'Scale'
]


def tta(model, image, tta_config):
    trans = er.MultiTransform(
        *tta_config
    )
    images = trans.transform(image)
    with torch.no_grad():
        outs = [model(im) for im in images]

    outs = trans.inv_transform(outs)

    out = sum(outs) / len(outs)

    return out


class TestTimeAugmentation(nn.Module):
    def __init__(self, module, tta_config):
        super(TestTimeAugmentation, self).__init__()
        self.module = module
        self.trans = er.MultiTransform(
            *tta_config
        )

    @torch.no_grad()
    def forward(self, image):
        images = self.trans.transform(image)
        outs = [self.module(im) for im in images]

        outs = self.trans.inv_transform(outs)

        out = sum(outs) / len(outs)
        return out




import numpy as np
import torch
import torch.nn.functional as F

from ever.interface.transform_base import Transform


class Identity(Transform):
    def transform(self, inputs):
        return inputs

    def inv_transform(self, transformed_inputs):
        return transformed_inputs


class Rotate90k(Transform):
    def __init__(self, k=1):
        super(Rotate90k, self).__init__()
        assert k in [1, 2, 3]

        self.k = k

    def transform(self, inputs):
        transformed_inputs = torch.rot90(inputs, self.k, [2, 3])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.rot90(transformed_inputs, 4 - self.k, [2, 3])
        return inputs


class HorizontalFlip(Transform):
    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.flip(inputs, [3])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [3])
        return inputs


class VerticalFlip(Transform):
    def __init__(self):
        super(VerticalFlip, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.flip(inputs, [2])
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.flip(transformed_inputs, [2])
        return inputs


class Transpose(Transform):
    def __init__(self):
        super(Transpose, self).__init__()

    def transform(self, inputs):
        transformed_inputs = torch.transpose(inputs, 2, 3)
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        inputs = torch.transpose(transformed_inputs, 2, 3)
        return inputs


class Scale(Transform):
    def __init__(self, size=None, scale_factor=None):
        super(Scale, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.input_shape = None

    def transform(self, inputs):
        self.input_shape = inputs.shape
        transformed_inputs = F.interpolate(inputs, size=self.size, scale_factor=self.scale_factor, mode='bilinear',
                                           align_corners=True)
        return transformed_inputs

    def inv_transform(self, transformed_inputs):
        size = (self.input_shape[2], self.input_shape[3])
        inputs = F.interpolate(transformed_inputs, size=size, mode='bilinear',
                               align_corners=True)
        return inputs


if __name__ == '__main__':
    # unit test
    Transform.unit_test(Rotate90k(k=1))
    Transform.unit_test(Rotate90k(k=2))
    Transform.unit_test(Rotate90k(k=3))

    Transform.unit_test(HorizontalFlip())
    Transform.unit_test(VerticalFlip())
    Transform.unit_test(Transpose())

    for scale_factor in np.linspace(0.25, 2.0, num=int((2.0 - 0.25) / 0.25 + 1)):
        Transform.unit_test(Scale(scale_factor=float(scale_factor)))

        Transform.unit_test(Scale(scale_factor=float(0.49)))

        Transform.unit_test(Scale(size=(894, 896)))
