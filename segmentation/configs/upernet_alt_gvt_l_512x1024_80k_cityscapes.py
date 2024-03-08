_base_ = [
    '_base_/models/upernet_r50.py', '_base_/datasets/cityscapes.py',
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


optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2,
            workers_per_gpu=5,
)