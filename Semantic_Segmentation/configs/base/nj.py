from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import ever as er

data = dict(
    train=dict(
        type='NJLoader',
        params=dict(
            image_dir=[
                './LoveDA/Qinhuai/images_png/',
                './LoveDA/Gulou/images_png/',
                './LoveDA/Qixia/images_png/',
                './LoveDA/Jianghan/images_png/',
                './LoveDA/Pukou/images_png/',
                './LoveDA/Lishui/images_png/',
                './LoveDA/Gaochun/images_png/',
                './LoveDA/Jiangxia/images_png/',
            ],
            mask_dir=[
                './LoveDA/Qinhuai/masks_png/',
                './LoveDA/Gulou/masks_png/',
                './LoveDA/Qixia/masks_png/',
                './LoveDA/Jianghan/masks_png/',
                './LoveDA/Pukou/masks_png/',
                './LoveDA/Lishui/masks_png/',
                './LoveDA/Gaochun/masks_png/',
                './LoveDA/Jiangxia/masks_png/',
            ],
            
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
            batch_size=16,
            num_workers=2,
        ),
    ),
    test=dict(
        type='NJLoader',
        params=dict(
            image_dir=[
                './LoveDA/Yuhuatai/images_png/',
                './LoveDA/Liuhe/images_png/',
                './LoveDA/Jintan/images_png/',
                 './LoveDA/Huangpi/images_png/',

            ],
            mask_dir=[
                './LoveDA/Yuhuatai/masks_png/',
                './LoveDA/Liuhe/masks_png/',
                './LoveDA/Jintan/masks_png/',
                './LoveDA/Huangpi/masks_png/',
            ],
            transforms=Compose([
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=0,
        ),
    ),
)
optimizer = dict(
    type='sgd',
    params=dict(
        momentum=0.9,
        weight_decay=0.0001
    ),
    grad_clip=dict(
        max_norm=35,
        norm_type=2,
    )
)
learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=0.01,
        power=0.9,
        max_iters=15000,
    ))
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=1000,
    eval_interval_epoch=20,
)

test = dict(

)
