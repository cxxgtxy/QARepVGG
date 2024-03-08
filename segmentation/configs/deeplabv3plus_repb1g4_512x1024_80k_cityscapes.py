_base_ = [
    '_base_/models/deeplabv3plus_r50-d8.py',
    '_base_/datasets/cityscapes.py', '_base_/default_runtime.py',
    '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RepVGGB1g4.pth',
    backbone=dict(
        _delete_=True,
        type='RepVGGB1g4'),
    decode_head=dict(
        c1_in_channels=128),
    auxiliary_head=dict(
        in_channels=512,)
    )

checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=3)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)