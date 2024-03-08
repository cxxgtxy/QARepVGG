_base_ = [
    './upernet_uniformer_s_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrained/uniformer_t.pth',
    backbone=dict(
        _delete_=True,
        type='uniformer_t'),
)
