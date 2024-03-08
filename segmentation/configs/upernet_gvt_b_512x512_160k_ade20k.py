_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k_lr.py'
]
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/gvt_base.pth',
    backbone=dict(
        type='gvt_base',
        style='pytorch'),
    decode_head=dict(num_classes=150, in_channels=[64, 160, 320, 640]),
    auxiliary_head=dict(num_classes=150,in_channels=320)
)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)