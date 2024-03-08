_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/slimformerv2_t_50.pth',
    backbone=dict(
        _delete_=True,
        type='slimformerv2_t_50_mid'),

    decode_head=dict(num_classes=150, in_channels=[96*4, 192*4, 384*4, 768*4]),
    auxiliary_head=dict(num_classes=150, in_channels=384*4)

)

