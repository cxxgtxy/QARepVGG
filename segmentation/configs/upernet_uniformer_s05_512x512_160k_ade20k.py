_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/uniformer_s_k05.pth',
    backbone=dict(
        _delete_=True,
        type='uniformer_s_k05'),
)


