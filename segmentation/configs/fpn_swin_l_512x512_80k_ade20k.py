_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/SwinL.pth',
    backbone=dict(
        type='SwinL',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150),
    )

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
