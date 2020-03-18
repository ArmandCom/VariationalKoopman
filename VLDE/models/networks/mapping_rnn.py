import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pyro
import pyro.distributions as dist

from models.networks.encoder import ImageEncoder

class ManifoldEncoder(nn.Module):
  def __init__(self, n_frames_input, n_frames_output, n_channels, image_size,
               feat_latent_size, time_enc_size, ngf, t_enc_rnn_hidden_size, trans_rnn_hidden_size, manifold_size):

    super(ManifoldEncoder, self).__init__()
    n_layers = int(np.log2(image_size)) - 1

    self.image_encoding_flag = False # Wether we work with videos.
    self.full_time_enc_flag = False # Wether we use all hidden vectors for the time encoding.

    if self.image_encoding_flag:
      # Option 1: DDPAE image encoder.
      self.image_encoder = ImageEncoder(n_channels, feat_latent_size, ngf, n_layers)
      # Option 2: If we use resnetnet as feature extractor
      # pretrained_resnet = models.resnet18(pretrained=True)
      # self.image_encoder = nn.Sequential(*list(pretrained_resnet.children())[:-1])
      # Option 3: Good encoder for MNIST
      # self.image_encoder = ImageEncoder([1, image_size, image_size], feat_latent_size)
      # Option 4: Pointnet++
      # Option 5: ELSE: no feature extraction for toy cases

    # Time encoding
    self.time_enc_rnn = nn.GRU(feat_latent_size, t_enc_rnn_hidden_size,
                               num_layers=1, batch_first=True, bidirectional=True)
    if self.full_time_enc_flag:
      self.time_enc_fc = nn.Linear(t_enc_rnn_hidden_size * 2 * n_frames_input, time_enc_size)
    else:
      self.time_enc_fc = nn.Linear(t_enc_rnn_hidden_size * 2, time_enc_size)

    # Transition rnn
    self.trans_rnn = nn.LSTMCell(feat_latent_size + time_enc_size + manifold_size,
                                 trans_rnn_hidden_size)  # TODO: change to LSTM similar to DDPAE (why cell?)
    self.y_mu = nn.Linear(trans_rnn_hidden_size, manifold_size)
    self.y_sigma = nn.Linear(trans_rnn_hidden_size, manifold_size)

    # Initial conditions
    # Option 1: RNN + fc
    # self.y0_rnn = nn.LSTM(feat_latent_size + time_enc_size, trans_rnn_hidden_size,
    #                                 num_layers=1, batch_first=True)
    # self.y0_mu = nn.Linear(trans_rnn_hidden_size, manifold_size)
    # self.y0_sigma = nn.Linear(trans_rnn_hidden_size, manifold_size)
    # Option 2: fc
    self.y0_mu = nn.Linear(feat_latent_size + time_enc_size, manifold_size)
    self.y0_sigma = nn.Linear(feat_latent_size + time_enc_size, manifold_size)

    # Prior encoder (backwards prediction): Note: We ignore it for this version.
    # self.n_prior = 0
    # self.n_window = 9
    # self.prior_rnn = nn.LSTMCell(manifold_size, trans_rnn_hidden_size)
    # self.prior_fc = nn.Linear(trans_rnn_hidden_size, manifold_size)

    self.input_size = image_size
    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.feat_latent_size = feat_latent_size
    self.time_enc_size = time_enc_size
    self.t_enc_rnn_hidden_size = t_enc_rnn_hidden_size
    self.trans_rnn_hidden_size = trans_rnn_hidden_size
    self.manifold_size = manifold_size

  def get_initial_pose(self, repr):
    '''
    Get initial pose of each component.
    '''
    # Repeat first input representation.
    output, _ = self.initial_pose_rnn(repr)
    output = output.contiguous().view(-1, self.hidden_size)
    initial_mu = self.initial_pose_mu(output)
    initial_sigma = self.initial_pose_sigma(output)
    initial_sigma = F.softplus(initial_sigma)
    return initial_mu, initial_sigma

  def encode(self, input, sample):


    batch_size, n_frames_input, n_channels, n_dimx, n_dimy = input.size()

    # Option 1: Raw input
    self.feat_latent_size = n_dimy
    input_repr = input.view(batch_size, n_frames_input, self.feat_latent_size)

    # Option 2: If input is a video
    # self.feat_latent_size = n_dim
    # input_repr = self.image_encoder(input.unsqueeze(2).view(-1, 1, n_dimx, n_dimy))#.repeat(1,3,1,1)

    '''TIME ENCODING'''
    input_repr = input_repr.view(batch_size, n_frames_input, -1)
    full_time_enc, time_enc = self.time_enc_rnn(input_repr)
    if self.full_time_enc_flag:
      time_enc = full_time_enc #Reverse backwards?
    time_enc = self.time_enc_fc(time_enc.permute(1, 0, 2).contiguous().view(batch_size, -1))


    '''MANIFOLD EMBEDDING'''

    input0 = torch.cat([input_repr[:, 0], time_enc], dim=-1)
    y0_mu = self.y0_mu(input0)
    y0_sigma = self.y0_sigma(input0)
    y0_sigma = F.softplus(y0_sigma)
    # Note: Activation? LeakyReLU
    y0 = self.pyro_sample('y_0', dist.Normal, y0_mu, y0_sigma, sample)
    y = y0
    ys = [y]

    # TODO: Try for a first version, but change LSTMCell to LSTM in v2 and input hidden=None
    # Note: If y0 has a RNN we can input the hidden space
    h, c = torch.zeros((batch_size, self.trans_rnn_hidden_size)).cuda(), \
           torch.zeros((batch_size, self.trans_rnn_hidden_size)).cuda()

    for i in range(1,n_frames_input):

        # With the time encoding, previous output and input features, obtain
        rnn_input = torch.cat([input_repr[:, i], y, time_enc], dim=1)
        h, c = self.trans_rnn(rnn_input, (h, c))

        # Obtain distribution parameters
        y_mu = self.y_mu(h)
        y_sigma = self.y_sigma(h)
        y_sigma = F.softplus(y_sigma)

        # Sample
        y = self.pyro_sample('y_%d'%{i}, dist.Normal, y_mu, y_sigma, sample)
        ys.append(y)

    # Note: Another option, would be to not sample at each iteration and do it at the end.
    # Note: Prior module has been removed. Backwards prediction using regressor instead.

    man = torch.stack(ys, dim=1)
    return man

  def forward(self, input, sample):
    '''
    param input: video of size (batch_size, n_frames_input, n_channels, dimx, dimy)
    Output: M
    '''
    M = self.encode(input, sample)

    return M
