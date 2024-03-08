_base_ = [
    '_base_/models/fcn_r50-d8.py', '_base_/datasets/ade20k_r4.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/QARepVGGA0.pth',
    backbone=dict(
        _delete_=True,
        type='QARepVGGA0'),
    decode_head=dict(
        in_channels=1280, num_classes=150),
    auxiliary_head=dict(
        in_channels=192, num_classes=150)
    )


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)