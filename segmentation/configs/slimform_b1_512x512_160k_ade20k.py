_base_ = [
    '_base_/models/simple_twins.py', '_base_/datasets/ade20k.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        _delete_=True,
        type='slimformer_b1_light_k05'),
    decode_head=dict(in_channels=[120, 240, 960, 1920], channels=768, dropout_ratio=0.0, num_classes=150),
    auxiliary_head=None,
    pretrained=None
)

optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
fp16 = dict()

optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.), 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))


lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

