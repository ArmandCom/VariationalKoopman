import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from typing import Tuple

from models.networks.model_pieces.base import BaseModule
from models.networks.model_pieces.blocks_2d import DownsampleBlock

# class ImageEncoder(nn.Module):
#   '''
#   Encodes images. Similar structure as DCGAN.
#   '''
#   def __init__(self, n_channels, output_size, ngf, n_layers):
#     super(ImageEncoder, self).__init__()
#
#     layers = [nn.Conv2d(n_channels, ngf, 4, 2, 1, bias=False),
#               nn.LeakyReLU(0.2, inplace=True)]
#
#     for i in range(1, n_layers - 1):
#       layers += [nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
#                  nn.BatchNorm2d(ngf * 2),
#                  nn.LeakyReLU(0.2, inplace=True)]
#       ngf *= 2
#
#     layers += [nn.Conv2d(ngf, output_size, 4, 1, 0, bias=False)]
#     # layers += [nn.Conv2d(ngf, ngf * 2, 4, 1, 0, bias=False),
#     #            nn.BatchNorm2d(ngf * 2),
#     #            nn.LeakyReLU(0.2, inplace=True)]
#     #
#     # ngf *= 2
#     #
#     # layers += [nn.Conv2d(ngf, output_size, 3, 1, 0, bias=False)]
#
#     self.main = nn.Sequential(*layers)
#
#
#
#   def forward(self, x):
#     # Note: input: [640 , 1, 64, 64], ouptut = [640 , 256] //squeezed[... , 1, 1] -> pose
#     x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
#     x = self.main(x)
#     x = x.squeeze(3).squeeze(2)
#     return x


class ImageEncoder(BaseModule):
  """Mnist encoder"""

  def __init__(self, input_shape, code_length):
    super(ImageEncoder, self).__init__()

    self.input_shape = input_shape
    self.code_length = code_length

    c, h, w = input_shape  # 1, 28, 28

    activation_fn = nn.LeakyReLU()

    # Convolutional network
    self.conv = nn.Sequential(
      DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
      DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn),
    )
    self.deepest_shape = (64, h // 4, w // 4)  # 64, 7, 7

    # FC network
    self.fc = nn.Sequential(
      nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
      nn.BatchNorm1d(num_features=64),
      activation_fn,
      nn.Linear(in_features=64, out_features=code_length),
      nn.Sigmoid()
    )

  def forward(self, x):
    # types: (torch.Tensor) -> torch.Tensor
    """
    Forward propagation.
    :param x: the input batch of images.
    :return: the batch of latent vectors.
    """
    h = x
    h = self.conv(h)
    h = h.view(len(h), -1)
    o = self.fc(h)  # o is 1, 64

    return o