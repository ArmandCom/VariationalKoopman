import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from models.networks.img_decoder import ImageDecoder
import sys
# sys.path.append("/home/ppalau/moments-vae/")

from functools import reduce
from operator import mul
from typing import Tuple

from models.networks.model_pieces.base import BaseModule
from models.networks.model_pieces.blocks_2d import UpsampleBlock

# class Decoder(nn.Module):
#   '''
#   Decode images from vectors. Similar structure as DCGAN.
#   '''
#   def __init__(self, input_size, n_channels, ngf, n_layers, activation='tanh'): #Note: is it tanh for sure?
#     super(Decoder, self).__init__()
#     input_size = 512
#     ngf = ngf * (2 ** (n_layers - 2))
#     self.ngf = ngf
#     layers = [nn.ConvTranspose2d(input_size, ngf, 3, 1, 0, bias=False),
#               nn.BatchNorm2d(ngf),
#               nn.LeakyReLU(True)]
#
#     layers += [nn.ConvTranspose2d(ngf, ngf//2, 4, 1, 0, bias=False),
#               nn.BatchNorm2d(ngf//2),
#               nn.LeakyReLU(True)]
#     ngf = ngf // 2
#
#     # layers = [nn.ConvTranspose2d(input_size, ngf, 4, 1, 0, bias=False),
#     #           nn.BatchNorm2d(ngf),
#     #           nn.ReLU(True)]
#
#     for i in range(1, n_layers - 1):
#       layers += [nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
#                  nn.BatchNorm2d(ngf // 2),
#                  nn.LeakyReLU(True)]
#       ngf = ngf // 2
#
#     layers += [nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False)]
#     if activation == 'tanh':
#       layers += [nn.Tanh()]
#     elif activation == 'sigmoid':
#       layers += [nn.Sigmoid()]
#     else:
#       raise NotImplementedError
#
#     self.main = nn.Sequential(*layers)
#
#   def forward(self, x):
#     # n_channels = x.shape[2]
#     # x = x.view(-1, n_channels, 1, 1)
#     x = self.main(x)
#     # x.register_hook(print)
#     return x


class Decoder(BaseModule):
    """
    MNIST model decoder.
    """
    def __init__(self, code_length, deepest_shape, output_shape):
        # type: (int, Tuple[int, int, int], Tuple[int, int, int]) -> None
        """
        Class constructor.
        :param code_length: the dimensionality of latent vectors.
        :param deepest_shape: the dimensionality of the encoder's deepest convolutional map.
        :param output_shape: the shape of MNIST samples.
        """
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.deepest_shape = deepest_shape
        self.output_shape = output_shape

        activation_fn = nn.LeakyReLU()

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, deepest_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, deepest_shape)),
            activation_fn
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.
        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o
