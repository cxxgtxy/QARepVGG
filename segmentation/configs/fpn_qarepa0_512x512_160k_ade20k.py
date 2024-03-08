_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/QARepVGGA0.pth',
    backbone=dict(
        _delete_=True,
        type='QARepVGGA0'),
    neck=dict(
        type='FPN',
        in_channels=[48, 96, 192, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
    )

data = dict(samples_per_gpu=2,
            workers_per_gpu=5,
)

checkpoint_config = dict(by_epoch=False, interval=16000)