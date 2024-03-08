_base_ = [
    'fpn_swin_t_512x512_80k_ade20k.py'
]

model = dict(
    pretrained='pretrained/uniformer_t_k05.pth',
    backbone=dict(
        _delete_=True,
        type='uniformer_t_k05'),
    )

