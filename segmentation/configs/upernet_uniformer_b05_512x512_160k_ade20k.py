_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/uniformer_b_k05.pth',
    backbone=dict(
        _delete_=True,
        type='uniformer_b_k05'),
    decode_head=dict(num_classes=150, in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(num_classes=150, in_channels=512)
)
