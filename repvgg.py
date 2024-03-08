import torch.nn as nn
import numpy as np
import torch
import copy
from se_block import SEBlock
from utils import my_scaler
import torch.nn.init as init
from functools import partial
from timm.models.layers import DropPath


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn_noaffline(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels, affine=False))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.use_scale = use_scale
        self.drop_path = DropPath(drop_path_ratio)

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2
        if isinstance(act, (type, )):
            if act == nn.PReLU:
                self.nonlinearity = nn.PReLU(num_parameters=out_channels)
            else:
                self.nonlinearity = act()
        else:
            self.nonlinearity = act

        # self.nonlinearity = act if not isinstance(act, (type,)) else act()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        if self.rbr_identity is None:
            return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(self.se(self.drop_path(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight if branch.bn.weight is not None else 1
            beta = branch.bn.bias if branch.bn.bias is not None else 0
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight if hasattr(branch, 'weight') else 1
            beta = branch.bias if hasattr(branch, 'bias') else 0
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class GRERepVGGBlock(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False):
        super(GRERepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale)
        del self.rbr_1x1
        del self.rbr_identity
        self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=True)
        self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self.bn = nn.BatchNorm2d(out_channels)

        # conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
        #                        padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        scale = my_scaler.get_scale()
        # scale = 1.0

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + scale*(self.bn(self.rbr_1x1(inputs) + id_out))))


class GRERepVGGBlockV0(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False):
        super(GRERepVGGBlockV0, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
        # del self.rbr_1x1
        # del self.rbr_identity
        # self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
        #                          stride=stride, padding=1, groups=groups, bias=False)
        # self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
        # self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        # self.bn1 = nn.BatchNorm2d(out_channels)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
        #                        padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        scale = my_scaler.get_scale()

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + scale*(self.rbr_1x1(inputs) + id_out))))


class QARepVGGBlockV1(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV2(RepVGGBlock):
    """
    only add bn for 3*3 before fusion. This is the default implementation for QARepVGG.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU(), drop_path_ratio=0.0):
        super(QARepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act, drop_path_ratio)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
            return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

        else:
            id_out = self.rbr_identity(inputs)
            return self.nonlinearity(self.bn(self.se(self.drop_path(self.rbr_dense(inputs) + self.rbr_1x1(inputs)) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        # kernel1x1, bias1x1 = self.rbr_1x1.weight, self.rbr_1x1.bias #self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        # kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        # kernel, bias = self._f

        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        tmp = K3 * t3
        l2_loss_circle = (tmp ** 2).sum() - (tmp[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1                         # The equivalent resultant central point of 3x3 kernel.
        if self.rbr_identity is not None:
            eq_kernel = eq_kernel + self.get_id_kernel()
        l2_loss_eq_kernel = (eq_kernel ** 2).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_id_kernel(self):
        if self.rbr_identity is not None:
            if self._id_tensor is None:
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 0, 0] = 1
                id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
                self._id_tensor = id_tensor
                return self._id_tensor
            else:
                return self._id_tensor
        return self._id_tensor


class QARepVGGBlockV3(RepVGGBlock):
    """
    only add bn for 1*1 before fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, bias=False, padding=1)
            # self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))


    def get_equivalent_kernel_bias(self):
        # kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)
        kernel3x3, bias3x3 = self.rbr_dense.weight + kernel1x1, bias1x1


        # kernel1x1, bias1x1 = self.rbr_1x1.weight, self.rbr_1x1.bias #self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        # kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel3x3 = kernel3x3 + id_tensor

        return self._fuse_extra_bn_tensor(kernel3x3, bias3x3, self.bn)


    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV4(RepVGGBlock):
    """
    only add bn for identity before fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV4, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, bias=False, padding=1)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            # self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))


    def get_equivalent_kernel_bias(self):
        kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        kernel = kernel + kernelid
        bias = biasid
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)


    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV5(RepVGGBlock):
    """
    add bn for 1x1 and 3*3 before fusion.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV5, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.bn = nn.BatchNorm2d(out_channels)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
            kernel = kernel + id_tensor
        return self._fuse_extra_bn_tensor(kernel, bias, self.bn)

    def switch_to_deploy(self):
        super(QARepVGGBlockV5, self).switch_to_deploy()
        if hasattr(self, 'bn'):
            self.__delattr__('bn')


class QARepVGGBlockV6(RepVGGBlock):
    """
    add bn for 1x1 and 3*3 before fusion, post BN is removed
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV6, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
            kernel = kernel + id_tensor
        return  kernel, bias


class QARepVGGBlockV6CL2(RepVGGBlock):
    """
    add bn for 1x1 and 3*3 before fusion, post BN is removed. Use custom L2
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV6CL2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
            kernel = kernel + id_tensor
        return  kernel, bias


class QARepVGGBlockM3(RepVGGBlock):
    """
    add bn for 1x1 and 3*3 before fusion, post BN is removed. Use custom L2
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockM3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self.rbr_1x1.weight, 0
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel+kernelid, bias+biasid

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = t3 / t3
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1                            # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


class QARepVGGBlockM3V2(QARepVGGBlockM3):
    """
    add bn for 1x1 and 3*3 before fusion, post BN is removed. Use custom L2
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockM3V2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        # t1 = t3 / t3
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1                            # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


class QARepVGGBlockV14(RepVGGBlock):
    """
    add bn for 1x1 before fusion, post BN is removed
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV14, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, bias=False, padding=1)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self.rbr_dense.weight, 0
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
            kernel = kernel + id_tensor
        return  kernel, bias


class QARepVGGBlockV15(RepVGGBlock):
    """
    add bn for 1x1 before fusion, post BN is removed
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV15, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        padding_11 = padding - kernel_size // 2
        if not deploy:
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.rbr_1x1 = conv_bn_noaffline(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
        else:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(kernel.device)
            kernel = kernel + id_tensor
        return  kernel, bias


class QARepVGGBlockV9(RepVGGBlock):
    """
    add bn for 1x1 and identity before fusion, post BN is removed
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV9, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_dense = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, groups=groups, bias=True, padding=1)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self.rbr_dense.weight, self.rbr_dense.bias
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid


class QARepVGGBlockV10(RepVGGBlock):
    """
    only add bn for 3*3 before fusion, remove BN
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV10, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3

        # kernel1x1, bias1x1 = self.rbr_1x1.weight, self.rbr_1x1.bias #self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # kernel, bias = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        # kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        # kernel, bias = self._f

        return kernel, bias

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV11(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV11, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale, act)

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle


class QARepVGGBlockV7(RepVGGBlock):
    """
    use three 3*3 branch.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV7, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                              padding_mode, deploy, use_se, use_scale, act)
        if not deploy:
            self.rbr_identity = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class QARepVGGBlock(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlock, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale, act)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            del self.rbr_1x1
            del self.rbr_identity
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=stride, padding=1, groups=groups, bias=False)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.bn = nn.BatchNorm2d(out_channels)


        # self.scales = nn.Parameter(torch.Tensor(3, out_channels, 1, 1))
        # init.constant_(self.scales[0], 0.25)
        # init.constant_(self.scales[1], 0.25)
        # init.constant_(self.scales[2], 0.5)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # scale = my_scaler.get_scale()
        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def _fuse_bn_tensor(self, kernel, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.weight.device)
            kernel = kernel + id_tensor
        kernel, bias = self._fuse_bn_tensor(kernel, self.bn)
        return kernel, bias
        # kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        # kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True


class QARepVGGBlockV8(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False, act=nn.ReLU()):
        super(QARepVGGBlockV8, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale, act)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            del self.rbr_1x1
            del self.rbr_identity
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=stride, padding=1, groups=groups, bias=True)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=True)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None


        # self.scales = nn.Parameter(torch.Tensor(3, out_channels, 1, 1))
        # init.constant_(self.scales[0], 0.25)
        # init.constant_(self.scales[1], 0.25)
        # init.constant_(self.scales[2], 0.5)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        # scale = my_scaler.get_scale()
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    def get_equivalent_kernel_bias(self):
        kernel = self.rbr_dense.weight + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_dense.weight.device)
            kernel = kernel + id_tensor
        # kernel, bias = self._fuse_bn_tensor(kernel, self.bn)
        return kernel, 0
        # kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        # kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        # return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=False)
        self.rbr_reparam.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        self.deploy = True


class GRERepVGGBlockV2(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False):
        super(GRERepVGGBlockV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale)
        if not deploy:
            del self.rbr_1x1
            del self.rbr_identity
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=stride, padding=1, groups=groups, bias=False)
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.bn = nn.BatchNorm2d(out_channels)

        # self.scales = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        # self.other_vector = nn.Parameter(torch.Tensor(2, out_channels, 1, 1))
        # init.constant_(self.scales, 0.25)
        # init.constant_(self.other_vector[0], 0.25)
        # init.constant_(self.other_vector[1], 0.5)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        scale = my_scaler.get_scale()

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + scale*(self.rbr_1x1(inputs) + id_out))))
        # return self.nonlinearity(self.bn(self.se(self.scales*self.rbr_dense(inputs) + scale*(self.other_vector[0]*self.rbr_1x1(inputs) + self.other_vector[1]*id_out))))

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel = self.rbr_dense.weight
        return self._fuse_extra_bn_tensor(kernel, 0, self.bn)


class GRERepVGGBlockV3(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False):
        super(GRERepVGGBlockV3, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale)
        if not deploy:
            del self.rbr_1x1
            del self.rbr_identity
            self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
            self.bn = nn.BatchNorm2d(out_channels)

        # self.scales = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        # self.other_vector = nn.Parameter(torch.Tensor(2, out_channels, 1, 1))
        # init.constant_(self.scales, 0.25)
        # init.constant_(self.other_vector[0], 0.25)
        # init.constant_(self.other_vector[1], 0.5)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        scale = my_scaler.get_scale()

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + scale * (self.rbr_1x1(inputs) + id_out))))


class GRERepVGGBlockV4(RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, use_scale=False):
        super(GRERepVGGBlockV4, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             padding_mode, deploy, use_se, use_scale)
        # del self.rbr_1x1
        # del self.rbr_identity
        # self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
        #                          stride=stride, padding=1, groups=groups, bias=False)
        # self.rbr_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=False)
        # self.rbr_identity = nn.Identity() if out_channels == in_channels and stride == 1 else None
        # self.bn = nn.BatchNorm2d(out_channels)

        # conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
        #                        padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        scale = my_scaler.get_scale()
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + scale*(self.rbr_1x1(inputs)) + id_out))


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, block_cls=RepVGGBlock, strides=[2, 2, 2, 2]):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = block_cls(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=strides[0], block_cls=block_cls)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=strides[1], block_cls=block_cls)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=strides[2], block_cls=block_cls)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=strides[3], block_cls=block_cls)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _init_weights(self, m):
        import math
        from timm.models.vision_transformer  import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def _make_stage(self, planes, num_blocks, stride, block_cls):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block_cls(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class RepVGGFeatures(RepVGG):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False, block_cls=RepVGGBlock, strides=[2, 2, 2, 2]):
        super(RepVGGFeatures, self).__init__(num_blocks, num_classes, width_multiplier, override_groups_map, deploy, use_se, block_cls, strides=strides)

    def _make_stage(self, planes, num_blocks, stride, block_cls):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(block_cls(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def init_weights(self, pretrained=None):
        import mmseg
        from mmseg.utils import get_root_logger
        from mmcv.runner import load_checkpoint
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        features = list()
        out = self.stage0(x)
        # features.append(out)
        out = self.stage1(out)
        features.append(out)
        out = self.stage2(out)
        features.append(out)
        out = self.stage3(out)
        features.append(out)
        out = self.stage4(out)
        features.append(out)
        # out = self.gap(out)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)

        return features


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}
A0_block = [2, 6, 20, 21]
gd_map = dict()
_i = 0
for l in optional_groupwise_layers:
    gd_map[l] = int(48*2**_i)
    if l in A0_block:
        _i += 1


def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A0_DW(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=gd_map, deploy=deploy)

def create_GRERepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_GRERepVGGV0_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlockV0)


def create_QARepVGGBlock_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlock)

def create_QARepVGGBlockV1_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV1)

def create_QARepVGGBlockV2_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)


def create_QARepVGGBlockV2_A0_d01(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=partial(QARepVGGBlockV2, drop_path_ratio=0.1))


def create_QARepVGGBlockV2_A0_DW(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=gd_map, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_QARepVGGBlockV3_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV3)

def create_QARepVGGBlockV4_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV4)

def create_QARepVGGBlockV5_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV5)

def create_QARepVGGBlockV6_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV6)

def create_QARepVGGBlockV6CL2_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV6CL2)

def create_QARepVGGBlockM3_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockM3)

def create_QARepVGGBlockM3V2_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockM3V2)

def create_QARepVGGBlockV7_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV7)


def create_QARepVGGBlockV8_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV8)

def create_QARepVGGBlockV9_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV9)

def create_QARepVGGBlockV10_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV10)

def create_QARepVGGBlockV11_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV11)

def create_QARepVGGBlockV14_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV14)

def create_QARepVGGBlockV15_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV15)

def create_QARepVGGBlock_RELU6_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=partial(QARepVGGBlock, act=nn.ReLU6()))

def create_QARepVGGBlock_LRELU_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=partial(QARepVGGBlock, act=nn.LeakyReLU()))

def create_QARepVGGBlock_PRELU_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=partial(QARepVGGBlock, act=nn.PReLU))

def create_QARepVGGBlockV2_PRELU_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=partial(QARepVGGBlockV2, act=nn.PReLU))

def create_GRERepVGGV2_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlockV2)

def create_GRERepVGGV3_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlockV3)

def create_GRERepVGGV4_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlockV4)


def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_GRERepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_QARepVGGBlockV2_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_GRERepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_QARepVGGBlockV2_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_QARepVGGBlock_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlock)

def create_QARepVGGBlockV2_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_QARepVGGBlock_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlock)

def create_QARepVGGBlockV2_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_QARepVGGBlockV2_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)

def create_GRERepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_QARepVGGBlockV2_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_QARepVGGBlockV2_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_GRERepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)

def create_QARepVGGBlockV2_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_GRERepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_GRERepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)

def create_GRERepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, block_cls=GRERepVGGBlock)

def create_RepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)

def create_QARepVGGBlockV2_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True, block_cls=QARepVGGBlockV2)

def create_GRERepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True, block_cls=GRERepVGGBlock)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A0-DW': create_RepVGG_A0_DW,
'GRERepVGG-A0':create_GRERepVGG_A0,
'GRERepVGGV0-A0':create_GRERepVGGV0_A0,
'QARepVGG-A0':create_QARepVGGBlock_A0,
'QARepVGGV1-A0':create_QARepVGGBlockV1_A0,
'QARepVGGV2-A0':create_QARepVGGBlockV2_A0,
'QARepVGGV2-A0_d01':create_QARepVGGBlockV2_A0_d01,
'QARepVGGV2-A0-DW':create_QARepVGGBlockV2_A0_DW,
'QARepVGGV3-A0':create_QARepVGGBlockV3_A0,
'QARepVGGV4-A0':create_QARepVGGBlockV4_A0,
'QARepVGGV5-A0':create_QARepVGGBlockV5_A0,
'QARepVGGV6-A0':create_QARepVGGBlockV6_A0,
'QARepVGGV6CL2-A0':create_QARepVGGBlockV6CL2_A0,
'QARepVGGBlockM3_A0':create_QARepVGGBlockM3_A0,
'QARepVGGBlockM3V2_A0':create_QARepVGGBlockM3V2_A0,
'QARepVGGV7-A0':create_QARepVGGBlockV7_A0,
'QARepVGGV8-A0':create_QARepVGGBlockV8_A0,
'QARepVGGV9-A0':create_QARepVGGBlockV9_A0,
'QARepVGGV10-A0':create_QARepVGGBlockV10_A0,
'QARepVGGV11-A0':create_QARepVGGBlockV11_A0,
'QARepVGGV14-A0':create_QARepVGGBlockV14_A0,
'QARepVGGV15-A0':create_QARepVGGBlockV15_A0,


'QARepVGG_RELU6-A0':create_QARepVGGBlock_RELU6_A0,
'QARepVGG_LRELU-A0':create_QARepVGGBlock_LRELU_A0,
'QARepVGG_PRELU-A0':create_QARepVGGBlock_PRELU_A0,
'QARepVGGV2PRELU-A0':create_QARepVGGBlockV2_PRELU_A0,
'GRERepVGGV2-A0':create_GRERepVGGV2_A0,
'GRERepVGGV3-A0':create_GRERepVGGV3_A0,
'GRERepVGGV4-A0':create_GRERepVGGV4_A0,

'RepVGG-A1': create_RepVGG_A1,
'QARepVGGV2-A1':create_QARepVGGBlockV2_A1,
'GRERepVGG-A1': create_GRERepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'GRERepVGG-A2': create_GRERepVGG_A2,
'QARepVGGV2-A2':create_QARepVGGBlockV2_A2,
'RepVGG-B0': create_RepVGG_B0,
'QARepVGG-B0':create_QARepVGGBlock_B0,
'QARepVGGV2-B0':create_QARepVGGBlockV2_B0,
'GRERepVGG-B0': create_GRERepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'QARepVGG-B1':create_QARepVGGBlock_B1,
'QARepVGGV2-B1':create_QARepVGGBlockV2_B1,
'GRERepVGG-B1': create_GRERepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'GRERepVGG-B1g2': create_GRERepVGG_B1g2,
'QARepVGGV2-B1g2': create_QARepVGGBlockV2_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'GRERepVGG-B1g4': create_GRERepVGG_B1g4,
'QARepVGGV2-B1g4': create_QARepVGGBlockV2_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'GRERepVGG-B2': create_GRERepVGG_B2,
'QARepVGGV2-B2':create_QARepVGGBlockV2_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'GRERepVGG-B2g2': create_GRERepVGG_B1g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'QARepVGGV2-B2g4': create_QARepVGGBlockV2_B2g4,
'GRERepVGG-B2g4': create_GRERepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'GRERepVGG-B3': create_GRERepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'GRERepVGG-B3g2': create_GRERepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
'GRERepVGG-B3g4': create_GRERepVGG_B3g4,
'RepVGG-D2se': create_RepVGG_D2se,     #   Updated at April 25, 2021. This is not reported in the CVPR paper.
'QARepVGGV2-D2se': create_QARepVGGBlockV2_D2se,
    #   Updated at April 25, 2021. This is not reported in the CVPR paper.
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]



#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

#   ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_pspnet = repvgg_model_convert(train_pspnet)
#   segmentation_test(deploy_pspnet)
#   =====================   example_pspnet.py shows an example

def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save({'model': model.state_dict()}, save_path)
    return model

