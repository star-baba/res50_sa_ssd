from torch import nn
from .vgg_base import VGG, conv_block
"""
https://stackoverflow.com/questions/55140554/convolutional-encoder-error-runtimeerror-input-and-target-shapes-do-not-matc/55143487#55143487

Here is the formula;

N --> Input Size, F --> Filter Size, stride-> Stride Size, pdg-> Padding size

ConvTranspose2d;

OutputSize = N*stride + F - stride - pdg*2

Conv2d;

OutputSize = (N - F)/stride + 1 + pdg*2/stride [e.g. 32/3=10 it ignores after the comma]
"""


class VGG11_bn(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv_block(1, input_channels, 64),
        
            *conv_block(1, 64, 128),

            *conv_block(2, 128, 256),

            *conv_block(2, 256, 512),

            *conv_block(2, 512, 512),
        ]

        super().__init__(model_name='vgg11_bn', conv_layers=nn.Sequential(*conv_layers), **kwargs)


class VGG11(VGG):
    def __init__(self, input_channels, **kwargs):
        conv_layers = [
            *conv_block(1, input_channels, 64, batch_norm=False),

            *conv_block(1, 64, 128, batch_norm=False),

            *conv_block(2, 128, 256, batch_norm=False),

            *conv_block(2, 256, 512, batch_norm=False),

            *conv_block(2, 512, 512, batch_norm=False),
        ]

        super().__init__(model_name='vgg11', conv_layers=nn.Sequential(*conv_layers), **kwargs)

