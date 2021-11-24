from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='FactSeg',
        params=dict(
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
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
