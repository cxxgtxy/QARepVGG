_base_ = [
    '_base_/models/fcn_r50-d8.py', '_base_/datasets/ade20k_r4.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='pretrained/resnet50_v1c-2cccc1ad.pth',
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))

checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=3)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6)