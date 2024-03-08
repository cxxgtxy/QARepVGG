_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/slimformerv2_s_50.pth',
    backbone=dict(
        _delete_=True,
        type='slimformerv2_s_50'),
)
