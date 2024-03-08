_base_ = [
    '_base_/models/deeplabv3plus_r50-d8.py',
    '_base_/datasets/cityscapes.py', '_base_/default_runtime.py',
    '_base_/schedules/schedule_40k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RepVGGA0.pth',
    backbone=dict(
        _delete_=True,
        type='RepVGGA0'),
    decode_head=dict(
        in_channels=1280,
        c1_in_channels=48),
    auxiliary_head=dict(
        in_channels=192,)
    )

checkpoint_config = dict(by_epoch=False, interval=8000, max_keep_ckpts=3)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)