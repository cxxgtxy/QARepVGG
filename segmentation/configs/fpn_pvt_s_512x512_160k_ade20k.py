_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]
model = dict(decode_head=dict(num_classes=150))

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/pvt_small.pth',
    backbone=dict(
        type='pvt_small',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    )

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)