_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/RepVGGB1g4.pth',
    backbone=dict(
        _delete_=True,
        type='RepVGGB1g4'),
    decode_head=dict(in_channels=[128, 256, 512, 2048], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150)
)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2,
            workers_per_gpu=5,
)

checkpoint_config = dict(by_epoch=False, interval=16000)
