# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='CosineAnnealing',  by_epoch=False,  warmup_iters=1500, min_lr_ratio=0.01)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')
