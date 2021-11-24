from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='FarSegV1Plus',
        params=dict(
            backbone=dict(
                type='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            classes=7,
            ppm=dict(
                in_channels=2048,
                pool_channels=512,
                out_channels=512,
                bins=(1, 2, 3, 6),
                bottleneck_conv='1x1',
                #dropout=0.1
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
                classifier_config=dict(
                    scale_factor=4.0,
                    num_classes=7,
                    kernel_size=3
                )
            ),
            loss=dict(
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
