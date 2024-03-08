_base_ = [
    '_base_/models/fcn_r50-d8.py', '_base_/datasets/cityscapes.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RepVGGA0.pth',
    backbone=dict(
        _delete_=True,
        type='RepVGGA0',
        strides=(2, 2, 1, 1),
    ),
    decode_head=dict(
        in_channels=1280),
    auxiliary_head=dict(
        in_channels=192,)
    )

checkpoint_config = dict(by_epoch=False, interval=16000)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)