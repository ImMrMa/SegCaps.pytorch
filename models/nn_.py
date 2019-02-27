import torch.utils.data
from torch.nn import functional as F

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn import ConvTranspose2d
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn as nn


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    # 修改这里的实现函数
    def forward(self, input):
        return conv2d_same(input, self.weight, self.bias, self.stride,
                           self.dilation, self.groups)

class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=False, dilation=1):
        super(ConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)

    def forward(self, input):
        input_size = input.size(2)
        output_size = input_size*self.stride[0]
        pad_l, pad_r = get_same(input_size,self.kernel_size[0],self.stride[0],dilation=1)
        #print(pad_l,pad_r)
        self.padding=max(pad_l,pad_r)
        input_size=(input_size-1)*self.stride[0]+self.kernel_size[0]-2*self.padding
        #print(input_size)
        output_padding=output_size-input_size
        #print(output_padding)
        return F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)



def conv2d_same(input, weight, bias=None, stride=[1, 1], dilation=(1, 1), groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                       (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.conv2d(input, weight, bias, stride,
                    padding=(padding_rows // 2, padding_cols // 2),
                    dilation=dilation, groups=groups)


def max_pool2d_same(input, kernel_size, stride=1, dilation=1, ceil_mode=False, return_indices=False):
    input_rows = input.size(2)
    out_rows = (input_rows + stride - 1) // stride
    padding_rows = max(0, (out_rows - 1) * stride +
                       (kernel_size - 1) * dilation + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    cols_odd = (padding_rows % 2 != 0)
    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])
    return F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding_rows // 2, dilation=dilation,
                        ceil_mode=ceil_mode, return_indices=return_indices)


def get_same(size, kernel, stride, dilation):
    out_size = (size + stride - 1) // stride
    padding = max(0, (out_size - 1) * stride +
                  (kernel - 1) * dilation + 1 - size)
    size_odd = (padding % 2 != 0)
    pad_l = padding // 2
    pad_r = padding // 2
    if size_odd:
        pad_l += 1
    return pad_l, pad_r
