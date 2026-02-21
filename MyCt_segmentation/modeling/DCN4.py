# --------------------------------------------------------
# Deformable Convolution v4
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
# from ..functions import DCNv4Function
import torch
import math
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

from table import TABLE, BWDTABLE
from DCNv4 import ext

def factors(N):
    res = []
    for i in range(1, N+1):
        if N % i == 0:
            res.append(i)
    return res

def findspec(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE:
        return TABLE[key][0], TABLE[key][1]

    d_stride = 8
    ms = factors(B*H*W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    key = f"{B}x{H}x{W}x{G}x{C}"
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread

def find_spec_bwd(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]

    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B*H*W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread

class DCNv4Function(Function):
    @staticmethod
    @custom_fwd
    def forward(
            ctx, input, offset_mask,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, dilation_h, dilation_w,
            group, group_channels, offset_scale,
            im2col_step, remove_center):

        forward_d_stride, forward_block_thread = findspec(input.shape[0], input.shape[1], input.shape[2], group, group_channels)
        backward_d_stride, backward_block_thread = find_spec_bwd(input.shape[0], input.shape[1], input.shape[2], group, group_channels)

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
            input, offset_mask, kernel_h,
            kernel_w, stride_h, stride_w, pad_h,
            pad_w, dilation_h, dilation_w, group,
            group_channels, offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        output = ext.dcnv4_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        args = [
            input, offset_mask, ctx.kernel_h,
            ctx.kernel_w, ctx.stride_h, ctx.stride_w, ctx.pad_h,
            ctx.pad_w, ctx.dilation_h, ctx.dilation_w, ctx.group,
            ctx.group_channels, ctx.offset_scale, ctx.im2col_step,
            grad_output.contiguous(), ctx.remove_center,
            ctx.backward_d_stride, ctx.backward_block_thread,
            False
        ]

        grad_input, grad_offset_mask = \
            ext.dcnv4_backward(*args)

        return grad_input, grad_offset_mask, \
            None, None, None, None, None, None, None,\
            None, None, None, None, None, None







class CenterFeatureScaleModule(nn.Module):
    def forward(self,
                query,
                center_feature_scale_proj_weight,
                center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale

class DCNv4(nn.Module):
    def __init__(
            self,
            channels=64,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=None,
            center_feature_scale=False,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs):
        """
        DCNv4 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group, ))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input, shape=None):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)


        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.view(N, H, W, C).permute(0, 3, 1, 2))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DCNv4Function.apply(
            x, offset_mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256,
            self.remove_center
            )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, L, -1)

        if not self.without_pointwise:
            x = self.output_proj(x)
        return x