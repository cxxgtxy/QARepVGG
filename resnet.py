from functools import partial


from timm.models import ResNet
from timm.models.resnet import Bottleneck, BasicBlock, _create_resnet, default_cfgs
import math
import copy
from timm.models.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, create_attn, create_classifier
from timm.models.registry import register_model
import torch
import torch.nn as nn
from timm.models.helpers import build_model_with_cfg

from utils import my_scaler


class GREBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(GREBottleneck, self).__init__(inplanes, planes, stride, downsample, cardinality, base_width,
                                            reduce_first, dilation, first_dilation, act_layer, norm_layer,
                                            attn_layer, aa_layer, drop_block, drop_path)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        scale = my_scaler.get_scale()
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        # if self.drop_path is not None:
        #     x = self.drop_path(x)

        if self.downsample is not None:
            residual = self.downsample(residual)
        x = scale * residual + x
        x = self.act3(x)

        return x


class GREBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(GREBasicBlock, self).__init__(inplanes, planes, stride, downsample, cardinality, base_width, reduce_first,
                                            dilation, first_dilation, act_layer, norm_layer, attn_layer, aa_layer,
                                            drop_block, drop_path)

    def forward(self, x):
        if self.downsample is not None:
            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            if self.drop_block is not None:
                x = self.drop_block(x)
            x = self.act1(x)
            if self.aa is not None:
                x = self.aa(x)

            x = self.conv2(x)
            x = self.bn2(x)
            if self.drop_block is not None:
                x = self.drop_block(x)

            if self.se is not None:
                x = self.se(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            if self.downsample is not None:
                residual = self.downsample(residual)
            x += residual
            x = self.act2(x)

            return x
        else:
            scale = my_scaler.get_scale()
            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            if self.drop_block is not None:
                x = self.drop_block(x)
            x = scale * residual + x
            x = self.act1(x)
            if self.aa is not None:
                x = self.aa(x)

            residual = x
            x = self.conv2(x)
            x = self.bn2(x)
            if self.drop_block is not None:
                x = self.drop_block(x)

            if self.se is not None:
                x = self.se(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            x += scale * residual
            x = self.act2(x)

            return x


class GREBasicBlockV2(BasicBlock):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
                 attn_layer=None, aa_layer=None, drop_block=None, drop_path=None):
        super(GREBasicBlockV2, self).__init__(inplanes, planes, stride, downsample, cardinality, base_width, reduce_first,
                                            dilation, first_dilation, act_layer, norm_layer, attn_layer, aa_layer,
                                            drop_block, drop_path)
        if self.downsample is None:
            first_planes = planes // reduce_first
            outplanes = planes * self.expansion
            self.conv_11 = nn.ModuleList()
            self.conv_11.append(nn.Sequential(nn.Conv2d(inplanes, first_planes, 1, bias=False), nn.BatchNorm2d(first_planes)))
            self.conv_11.append(nn.Sequential(nn.Conv2d(first_planes, outplanes, 1, bias=False), nn.BatchNorm2d(outplanes)))
            self.identity = nn.ModuleList()
            self.identity.append(nn.BatchNorm2d(first_planes))
            self.identity.append(nn.BatchNorm2d(outplanes))

    def forward(self, x):
        if self.downsample is not None:
            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            if self.drop_block is not None:
                x = self.drop_block(x)
            x = self.act1(x)
            if self.aa is not None:
                x = self.aa(x)

            x = self.conv2(x)
            x = self.bn2(x)
            if self.drop_block is not None:
                x = self.drop_block(x)

            if self.se is not None:
                x = self.se(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            if self.downsample is not None:
                residual = self.downsample(residual)
            x += residual
            x = self.act2(x)

            return x
        else:
            scale = my_scaler.get_scale()
            residual = x

            x = self.conv1(x)
            x = self.bn1(x)
            if self.drop_block is not None:
                x = self.drop_block(x)
            x = scale * (self.identity[0](residual) + self.conv_11[0](residual)) + x
            x = self.act1(x)
            if self.aa is not None:
                x = self.aa(x)

            residual = x
            x = self.conv2(x)
            x = self.bn2(x)
            if self.drop_block is not None:
                x = self.drop_block(x)

            if self.se is not None:
                x = self.se(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            x = scale * (self.identity[1](residual) + self.conv_11[1](residual)) + x
            x = self.act2(x)

            return x


def _create_gre_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)


@register_model
def gre_resnet50(pretrained=False, **kwargs):
    """Constructs a GRE ResNet-50 model.
    """
    model_args = dict(block=GREBottleneck, layers=[3, 4, 6, 3],  **kwargs)
    return _create_gre_resnet('resnet50', pretrained, **model_args)


@register_model
def gre_resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=GREBasicBlock, layers=[3, 4, 6, 3],  **kwargs)
    return _create_gre_resnet('resnet34', pretrained, **model_args)


@register_model
def gre_resnet34v2(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=GREBasicBlockV2, layers=[3, 4, 6, 3],  **kwargs)
    return _create_gre_resnet('resnet34', pretrained, **model_args)