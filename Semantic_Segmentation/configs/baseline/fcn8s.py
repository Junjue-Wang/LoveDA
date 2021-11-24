from configs.base.loveda import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='FCN8s',
        params=dict(
            pretrained=True,
            classes=7,
            loss=dict(
                ignore_index=-1,
                ce=dict(),
            ),
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)