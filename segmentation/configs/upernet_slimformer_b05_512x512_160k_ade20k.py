_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/slimformerv2_b_50.pth',
    backbone=dict(
        _delete_=True,
        type='slimformerv2_b_50'),
    decode_head=dict(num_classes=150, in_channels=[128, 256, 512, 1024]),
    auxiliary_head=dict(num_classes=150, in_channels=512)

)
