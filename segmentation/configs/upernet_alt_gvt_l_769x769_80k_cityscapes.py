_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/cityscapes_769x769.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='alt_gvt_large',
        style='pytorch'),
    decode_head=dict(in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(in_channels=512)
)


optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2,
            workers_per_gpu=5,
)