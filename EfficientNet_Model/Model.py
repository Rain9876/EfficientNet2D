from torch import nn
from .Utils import (
    BLOCKS_ARGS,
    EfficientSwish,
    round_filters,
    round_repeated,
    drop_connect,
    get_same_static_padding_conv2d,
    Flatten
)


# Sequeeze and Excitation layer
class Sequeeze_Excitation(nn.Module):
    def __init__(self, in_channels, reduced_channel):
        super().__init__()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.se_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_channel, kernel_size=1, padding=0, bias=True),
            EfficientSwish(),
            nn.Conv2d(in_channels=reduced_channel, out_channels=in_channels, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = self.se_layer(self.pooling(x))
        return x_se * x


# Mobile Inverted Residual Bottleneck Block With the SE layer
# Bottleneck architecture begins with 1x1 and ends with 1x1
class MBCov_Block(nn.Module):
    def __init__(self, block_args, batch_norm_momentum, batch_norm_epsilon, image_size):
        super().__init__()

        self.block_args = block_args

        Conv2d = get_same_static_padding_conv2d(image_size=image_size)

        # Pointwise Conv (Expand layer) k1x1
        expand_channels = block_args.input_channels * block_args.expand_ratio

        if block_args.expand_ratio > 1:
            self.expand_conv = nn.Sequential(
                Conv2d(in_channels=block_args.input_channels, out_channels=expand_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=expand_channels, momentum=batch_norm_momentum, eps=batch_norm_epsilon),
                EfficientSwish()
            )

        # Depthwise Conv (Through groups)
        self.depthwise_conv = nn.Sequential(
            Conv2d(in_channels=expand_channels, out_channels=expand_channels, kernel_size=block_args.kernel_size,
                   stride=block_args.strides, groups=expand_channels, bias=False),
            nn.BatchNorm2d(num_features=expand_channels, momentum=batch_norm_momentum, eps=batch_norm_epsilon),
            EfficientSwish(),
        )

        # Squeeze and Excitation layer
        if block_args.se_ratio:
            reduced_channel = max(1, int(block_args.input_channels * block_args.se_ratio))
            self.se_layer = Sequeeze_Excitation(expand_channels, reduced_channel)

        # Pointwise Conv (Expand layer) k1x1
        self.output_conv = nn.Sequential(
            Conv2d(in_channels=expand_channels, out_channels=block_args.output_channels, kernel_size=1, stride=1,
                   bias=False),
            nn.BatchNorm2d(num_features=block_args.output_channels, momentum=batch_norm_momentum,
                           eps=batch_norm_epsilon),
            EfficientSwish(),
        )

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs

        if self.block_args.expand_ratio > 1:
            x = self.expand_conv(x)

        x = self.depthwise_conv(x)
        if self.block_args.se_ratio > 0:
            x = self.se_layer(x)
        x = self.output_conv(x)

        # id skip, drop connect
        if self.block_args.id_skip and self.block_args.strides == 1 and \
                self.block_args.input_channels == self.block_args.output_channels:
            if drop_connect_rate and self.training:
                x = drop_connect(x, drop_connect_rate)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, model_name, width_coeff, depth_coeff, resolu, dropout_rate, depth_divisor, batch_norm_epsilon,
                 batch_norm_momentum, drop_connect_rate, num_output, blocks_args=BLOCKS_ARGS):
        super().__init__()

        batch_norm_momentum = 1 - batch_norm_momentum
        self.drop_connect_rate = drop_connect_rate
        self.blocks_args = blocks_args
        self.model_name = model_name
        self.archtecture = []

        Conv2d = get_same_static_padding_conv2d(image_size=resolu)

        # Stem
        out_channel = round_filters(32, width_coeff, depth_divisor)

        self.stem = nn.Sequential(
            Conv2d(3, out_channel, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=out_channel, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
            EfficientSwish()
        )

        layer = "Conv{:d}x{:d} in: {:d}, out:{:d}\n".format(3, 3, 3, out_channel)
        self.archtecture.append(layer)

        # Blocks
        self.blocks = nn.ModuleList([])  ## like the module link list, append layers
        for i, block_args in enumerate(blocks_args):

            block_args = block_args._replace(
                input_channels=round_filters(block_args.input_channels, width_coeff, depth_divisor),
                output_channels=round_filters(block_args.output_channels, width_coeff, depth_divisor),
                num_repeat=round_repeated(block_args.num_repeat, depth_coeff)
            )

            self.blocks.append(
                MBCov_Block(block_args, batch_norm_momentum, batch_norm_epsilon, resolu))

            layer = "MB_Conv{:d} {:d}x{:d} in: {:d}, out:{:d}\n".format(block_args.expand_ratio,
                                                                        block_args.kernel_size,
                                                                        block_args.kernel_size,
                                                                        block_args.input_channels,
                                                                        block_args.output_channels)
            self.archtecture.append(layer)

            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_channels=block_args.output_channels, strides=1)

            for _ in range(block_args.num_repeat - 1):
                self.blocks.append(
                    MBCov_Block(block_args, batch_norm_momentum, batch_norm_epsilon, resolu))

                layer = "MB_Conv{:d} {:d}x{:d} in: {:d}, out:{:d}\n".format(block_args.expand_ratio,
                                                                            block_args.kernel_size,
                                                                            block_args.kernel_size,
                                                                            block_args.input_channels,
                                                                            block_args.output_channels)
                self.archtecture.append(layer)

        # Head
        in_channel = block_args.output_channels
        out_channel = round_filters(1280, width_coeff, depth_divisor)
        self.head = nn.Sequential(
            Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channel, eps=batch_norm_epsilon, momentum=batch_norm_momentum),
            EfficientSwish(),
        )
        layer = "Conv{:d}x{:d} in: {:d}, out:{:d}\n".format(1, 1, in_channel, out_channel)
        self.archtecture.append(layer)

        # FC
        self.FC = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout2d(dropout_rate),
            nn.Linear(out_channel, num_output),
        )
        layer = "FC in: {:d}, out:{:d}\n".format(out_channel, num_output)
        self.archtecture.append(layer)

    def forward(self, inputs):

        x = self.stem(inputs)

        for i, block in enumerate(self.blocks):
            if self.drop_connect_rate:
                drop_connect_rate = self.drop_connect_rate * i / len(self.blocks)
                x = block(x, drop_connect_rate)

        x = self.head(x)

        x = self.FC(x)

        return x

    def model_structure(self):
        print(self.model_name)
        print()
        for idx, layer in enumerate(self.archtecture):
            print(str(idx) + "\t" + layer)


def EfficientNetB0(num_oup):
    return EfficientNet("efficientNet-B0", 1.0, 1.0, 224, 0.2, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB1(num_oup):
    return EfficientNet("efficientNet-B1", 1.0, 1.1, 240, 0.2, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB2(num_oup):
    return EfficientNet("efficientNet-B2", 1.1, 1.2, 260, 0.3, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB3(num_oup):
    return EfficientNet("efficientNet-B3", 1.2, 1.4, 300, 0.3, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB4(num_oup):
    return EfficientNet("efficientNet-B4", 1.4, 1.8, 380, 0.4, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB5(num_oup):
    return EfficientNet("efficientNet-B5", 1.6, 2.2, 456, 0.4, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB6(num_oup):
    return EfficientNet("efficientNet-B6", 1.8, 2.6, 528, 0.5, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)


def EfficientNetB7(num_oup):
    return EfficientNet("efficientNet-B7", 2.0, 3.1, 600, 0.5, depth_divisor=8, batch_norm_epsilon=0.001,
                        batch_norm_momentum=0.99, drop_connect_rate=0.2, num_output=num_oup)
