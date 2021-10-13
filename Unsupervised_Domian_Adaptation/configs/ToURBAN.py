from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er


TARGET_SET = 'URBAN'

source_dir = dict(
    image_dir=[
        './LoveDA/Pukou/images_png/',
        './LoveDA/Lishui/images_png/',
        './LoveDA/Gaochun/images_png/',
        './LoveDA/Jiangxia/images_png/',
        
    ],
    mask_dir=[
        './LoveDA/Pukou/masks_png/',
        './LoveDA/Lishui/masks_png/',
        './LoveDA/Gaochun/masks_png/',
        './LoveDA/Jiangxia/masks_png/',
    ],
)
target_dir = dict(
    image_dir=[
        './LoveDA/Jianye/images_png/',
        './LoveDA/Wuchang/images_png/',
        './LoveDA/Wujin/images_png/',
    ],
    mask_dir=[
        './LoveDA/Jianye/masks_png/',
        './LoveDA/Wuchang/masks_png/',
         './LoveDA/Wujin/masks_png/',
    ],
)


SOURCE_DATA_CONFIG = dict(
    image_dir=source_dir['image_dir'],
    mask_dir=source_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=2,
)


TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        RandomCrop(512, 512),
        OneOf([
            HorizontalFlip(True),
            VerticalFlip(True),
            RandomRotate90(True)
        ], p=0.75),
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=2,
)

EVAL_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=target_dir['mask_dir'],
    transforms=Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        er.preprocess.albu.ToTensor()

    ]),
    CV=dict(k=10, i=-1),
    training=False,
    batch_size=2,
    num_workers=0,
)
