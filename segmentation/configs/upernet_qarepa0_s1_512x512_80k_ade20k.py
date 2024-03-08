_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/QARepVGGA0.pth',
    backbone=dict(
        _delete_=True,
        type='QARepVGGA0',
        strides=(2, 2, 1, 1),

    ),
    decode_head=dict(in_channels=[48, 96, 192, 1280], num_classes=150),
    auxiliary_head=dict(in_channels=192, num_classes=150)
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=4,
            workers_per_gpu=5,
)

checkpoint_config = dict(by_epoch=False, interval=16000)
