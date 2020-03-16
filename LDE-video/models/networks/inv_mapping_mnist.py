import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from models.networks.img_decoder import ImageDecoder

# class Decoder(nn.Module):
#
#   def __init__(self, input_shape, manifold_shape, hidden_shape):
#     super(Decoder, self).__init__()
#
#     layers = [nn.Linear(manifold_shape, hidden_shape),
#               nn.ReLU(True)]
#
#     for i in range(1):
#       layers += [nn.Linear(hidden_shape, hidden_shape),
#                  nn.ReLU(True)] # nn.BatchNorm2d(hidden_shape),
#
#     layers += [nn.Linear(hidden_shape, input_shape)]
#
#     self.main = nn.Sequential(*layers)
#
#   def forward(self, input):
#     return self.main(input)
#

class Decoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, input_size, n_channels, ngf, n_layers, activation='tanh'): #Note: is it tanh for sure?
    super(Decoder, self).__init__()
    input_size = 512
    ngf = ngf * (2 ** (n_layers - 2))
    self.ngf = ngf
    layers = [nn.ConvTranspose2d(input_size, ngf, 3, 1, 0, bias=False),
              nn.BatchNorm2d(ngf),
              nn.LeakyReLU(True)]

    layers += [nn.ConvTranspose2d(ngf, ngf//2, 4, 1, 0, bias=False),
              nn.BatchNorm2d(ngf//2),
              nn.LeakyReLU(True)]
    ngf = ngf // 2

    # layers = [nn.ConvTranspose2d(input_size, ngf, 4, 1, 0, bias=False),
    #           nn.BatchNorm2d(ngf),
    #           nn.ReLU(True)]

    for i in range(1, n_layers - 1):
      layers += [nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(ngf // 2),
                 nn.LeakyReLU(True)]
      ngf = ngf // 2

    layers += [nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False)]
    if activation == 'tanh':
      layers += [nn.Tanh()]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    else:
      raise NotImplementedError

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    # n_channels = x.shape[2]
    # x = x.view(-1, n_channels, 1, 1)
    x = self.main(x)
    # x.register_hook(print)
    return x


