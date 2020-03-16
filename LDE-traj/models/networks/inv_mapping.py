import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.networks.img_decoder import ImageDecoder

class Decoder(nn.Module):

  def __init__(self, n_frames_input, n_frames_output, n_channels, image_size,
               feat_latent_size, ngf, manifold_size):
    super(Decoder, self).__init__()

    # n_layers = int(np.log2(image_size)) - 1
    # self.image_decoder = ImageDecoder(n_channels, feat_latent_size, ngf, n_layers)

    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.feat_latent_size = feat_latent_size
    self.manifold_size = manifold_size

  def decode(self, Y):
    return Y


  def forward(self, input):
    return self.decode(input)