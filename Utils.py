import math
import torch
from torch import nn
from collections import namedtuple
from functools import partial

"""
BlockArgs records the params required for 7 MB_blocks. 
    For example: In stage 2, MBCov1_k3x3 repeat only once, input resol 112x112, input chanel 32, output_channel 16
                 output resol 112x112 (expand_ratio = 1), stride 1.

num_repeat: the number of times that MB block repeated.
expand_ratio: Expansion for the the number of neuron in each layer.
id_skip: whether the layer is allowed to be skip. 
SE_ratio: Prepare for Squeeze and Excitation layer, the ratio for squeezing
"""

blockArgs = namedtuple("Block_args",
                       ["kernel_size", "num_repeat", "input_channels", "output_channels", "expand_ratio",
                        "id_skip", "se_ratio", "strides"])

blockArgs.__new__.__defaults__ = (None,) * len(blockArgs._fields)

BLOCK_ARGS = [
    blockArgs(kernel_size=3, num_repeat=1, input_channels=32, output_channels=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    blockArgs(kernel_size=3, num_repeat=2, input_channels=16, output_channels=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    blockArgs(kernel_size=5, num_repeat=2, input_channels=24, output_channels=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    blockArgs(kernel_size=3, num_repeat=3, input_channels=40, output_channels=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    blockArgs(kernel_size=5, num_repeat=3, input_channels=80, output_channels=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    blockArgs(kernel_size=5, num_repeat=4, input_channels=112, output_channels=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    blockArgs(kernel_size=3, num_repeat=1, input_channels=192, output_channels=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]


# Construct the swish activation function  to replace the ReLU6
# Swish f(x) = x * sigmoid(x) works better than ReLu in deeper model
# https://arxiv.org/abs/1710.05941

# A formula differentiate the Swish operation
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i + i * sigmoid_i * (1 - sigmoid_i))


# MemoryEfficientSwish saves more memories than just using swish
# Save the input for back propagation, save more space for computing derivative
# https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
class EfficientSwish(nn.Module):
    def forward(self, input):
        S = Swish.apply
        return S(input)


# The width of Network / the number of neurons of each CNN layer multiples width_coeff
# The number of neuron is the number of filter.
# Return the the the number of filter after width expansion.
def round_filters(filters, width_coeff, depth_divisor):
    filters *= width_coeff
    new_filters = int(
        filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


# The depth of Network / the repeated times of each block multiples depth_coeff
# For example, the block repeat 3 times, therefore, the real depth for the block is multiplied by depth coeff
def round_repeated(repeats, depth_coeff):
    return int(math.ceil(depth_coeff * repeats))


# DropConnect works similar to Dropout, except that drop connect disable individual weights (i.e., set them to zero),
# instead of nodes
def drop_connect(inputs, drop_connect_rate):
    batch_size = list(inputs.size())[0]
    keep_prob = 1.0 - drop_connect_rate
    mask = torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype) + keep_prob
    mask = torch.floor(mask)  # Either 1 or 0
    return inputs * mask / keep_prob


# Call same padding methods, given the image size
def get_same_static_padding_conv2d(image_size=None):
    return partial(Conv2dSamePadding, image_size=image_size)


# Custom same padding for the Cov2d for Pytorch
# Static same padding based on the fixed image size
class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, **kwargs):

        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2  # Force stride to be tuple

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)

        # Fill with Zero for the both left and right padding area
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))  ## Padding left and right
        else:
            ## No padding, leave input
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)  ## The padding image
        x = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


# Identity layer
# Return the output same with the input
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input
