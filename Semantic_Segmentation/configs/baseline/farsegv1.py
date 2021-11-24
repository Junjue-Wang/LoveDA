from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='FarSegV1',
        params=dict(
            backbone=dict(
                type='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            classes=7,
            
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
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
                ce=dict(
                ),
                tverloss=dict(
                    alpha=0.5,
                    beta=0.5,
                    scaler=1.0,
                )
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
