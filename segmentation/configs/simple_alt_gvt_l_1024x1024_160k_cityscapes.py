_base_ = [
    '_base_/models/simple_twins.py', '_base_/datasets/cityscapes_1024x1024.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/alt_gvt_large.pth',
    backbone=dict(
        type='alt_gvt_large',
        style='pytorch',
        drop_path_rate=0.2
    ),
    decode_head=dict(in_channels=[128, 256, 512, 1024], channels=768,),
    # auxiliary_head=dict(in_channels=512, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    auxiliary_head=None,
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768))

)


optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=1,
            workers_per_gpu=5,
)

checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=4000, metric='mIoU')