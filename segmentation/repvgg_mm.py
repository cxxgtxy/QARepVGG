import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from pathlib import  Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
# print(sys.path)
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
#from mmseg.utils import get_root_logger
#from mmcv.runner import load_checkpoint
# from timm.models.vision_transformer import Block as TimmBlock
# from timm.models.vision_transformer import Attention as TimmAttention


# import pvt
from repvgg import RepVGGFeatures, g4_map, QARepVGGBlockV2, RepVGGBlock


@BACKBONES.register_module()
class QARepVGGB1g4(RepVGGFeatures):
    def __init__(self, num_blocks=[4, 6, 16, 1], num_classes=1000, width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, 
                 deploy=False, block_cls=QARepVGGBlockV2, strides=[2, 2, 2, 2], **kwargs):
        super(QARepVGGB1g4, self).__init__(num_blocks, num_classes, width_multiplier, override_groups_map, deploy, block_cls=block_cls, strides=strides, **kwargs)


@BACKBONES.register_module()
class QARepVGGA0(RepVGGFeatures):
    def __init__(self, num_blocks=[2, 4, 14, 1], num_classes=1000, width_multiplier=[0.75, 0.75, 0.75, 2.5],
                 override_groups_map=None, deploy=False, block_cls=QARepVGGBlockV2, strides=[2, 2, 2, 2], **kwargs):
        super(QARepVGGA0, self).__init__(num_blocks, num_classes, width_multiplier, override_groups_map, deploy, block_cls=block_cls, strides=strides, **kwargs)


@BACKBONES.register_module()
class RepVGGB1g4(RepVGGFeatures):
    def __init__(self, num_blocks=[4, 6, 16, 1], num_classes=1000, width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map,
                 deploy=False, block_cls=RepVGGBlock, strides=[2, 2, 2, 2], **kwargs):
        super(RepVGGB1g4, self).__init__(num_blocks, num_classes, width_multiplier, override_groups_map, deploy, block_cls=block_cls, strides=strides, **kwargs)


@BACKBONES.register_module()
class RepVGGA0(RepVGGFeatures):
    def __init__(self, num_blocks=[2, 4, 14, 1], num_classes=1000, width_multiplier=[0.75, 0.75, 0.75, 2.5],
                 override_groups_map=None, deploy=False, block_cls=RepVGGBlock, strides=[2, 2, 2, 2], **kwargs):
        super(RepVGGA0, self).__init__(num_blocks, num_classes, width_multiplier, override_groups_map, deploy, block_cls=block_cls, strides=strides, **kwargs)