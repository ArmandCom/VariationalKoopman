import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.networks.img_encoder import ImageEncoder
from torchvision import models

class Encoder(nn.Module):

  def __init__(self, n_frames_input, n_frames_output, n_channels, image_size,
               feat_latent_size, time_enc_size, ngf, t_enc_rnn_hidden_size, trans_rnn_hidden_size, manifold_size):
    super(Encoder, self).__init__()

    input_size = image_size
    enc_hidden_size = int(input_size//2)
    n_layers = int(np.log2(image_size)) - 1
    # self.image_encoder = ImageEncoder(n_channels, feat_latent_size, ngf, n_layers)
    # pretrained_resnet = models.resnet18(pretrained=True)
    # self.image_encoder = nn.Sequential(*list(pretrained_resnet.children())[:-1])
    self.image_encoder = ImageEncoder([1, input_size, input_size], feat_latent_size)

    # Time encoding
    self.time_enc_rnn = nn.GRU(feat_latent_size, t_enc_rnn_hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)
    self.time_enc_fc = nn.Linear(t_enc_rnn_hidden_size * 2, time_enc_size)

    # Transition rnn
    self.trans_rnn = nn.LSTMCell(feat_latent_size + time_enc_size + manifold_size,
                                 trans_rnn_hidden_size)
    self.trans_fc = nn.Linear(trans_rnn_hidden_size, manifold_size)
    # Note: layer norm?

    # Prior encoder
    self.n_prior = 0
    self.n_window = 9
    self.prior_rnn = nn.LSTMCell(manifold_size, trans_rnn_hidden_size)
    self.prior_fc = nn.Linear(trans_rnn_hidden_size, manifold_size)

    # Initial conditions
    self.initial_cond_rnn = nn.LSTM(trans_rnn_hidden_size, trans_rnn_hidden_size,
                                    num_layers=1, batch_first=True)
    self.initial_cond_fc = nn.Linear(trans_rnn_hidden_size, manifold_size)

    self.input_size = input_size
    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.feat_latent_size = feat_latent_size
    self.time_enc_size = time_enc_size
    self.t_enc_rnn_hidden_size = t_enc_rnn_hidden_size
    self.trans_rnn_hidden_size = trans_rnn_hidden_size
    self.manifold_size = manifold_size

  def encode(self, input):


    batch_size, n_frames_input, chan, n_dimx, n_dimy = input.size()

    '''TIME ENCODING'''
    # self.feat_latent_size = n_dim# Note: future feature extractor

    input_repr = self.image_encoder(input.unsqueeze(2).view(-1, 1, n_dimx, n_dimy))#.repeat(1,3,1,1)
    # input_repr = input_repr.view(batch_size, n_frames_input, -1)
    # _, time_enc = self.time_enc_rnn(input_repr)
    # # time_enc = time_enc.view(2, batch_size, -1)
    #
    #
    # time_enc = self.time_enc_fc(time_enc.permute(1, 0, 2).contiguous().view(batch_size, -1))
    #
    # # # Instance normalization
    # # norm = nn.InstanceNorm2d(1)
    #
    # '''MANIFOLD EMBEDDING'''
    # ys = []
    # first_hidden_states = []
    # h, c = torch.zeros((batch_size, self.trans_rnn_hidden_size)).cuda(), \
    #        torch.zeros((batch_size, self.trans_rnn_hidden_size)).cuda()
    # y = torch.zeros(batch_size, self.manifold_size).cuda()
    # # ys.append(y)
    # for i in range(n_frames_input):
    #     rnn_input = torch.cat([input_repr[:, i, ...], y, time_enc], dim=1)
    #     h, c = self.trans_rnn(rnn_input, (h, c))
    #     y = self.trans_fc(h)
    #     if i == 0:
    #       #How many initial points?
    #       first_hidden_states.append(h.view(batch_size, 1, -1)) #Note: is it h or c?
    #     # Note: Layer norm and activation
    #     # Note: Save hidden states?
    #     ys.append(y)
    #
    # #TODO: prior as weighted sum of previous datapoints
    # # y_prior = []
    # # for n in range(self.n_window + self.n_prior, 0, -1):
    # #     rnn_input = ys[n]
    # #     h, c = self.prior_rnn(rnn_input, (h, c))
    # #     if n <= self.n_prior:
    # #       ys[n-1] = self.prior_fc(h)
    #
    # first_hidden_states = torch.cat(first_hidden_states, dim=1)
    # initial_cond = self.get_initial_cond(first_hidden_states)
    #
    # M = torch.stack(ys, dim=1) + initial_cond
    M = input_repr
    return M

  def get_initial_cond(self, repr):
    '''
    Get initial pose of each component.
    '''
    # Repeat first input representation.
    output, _ = self.initial_cond_rnn(repr)
    output = output.contiguous().view(-1, self.trans_rnn_hidden_size)
    initial_cond = self.initial_cond_fc(output).unsqueeze(1)

    return initial_cond

  def forward(self, input):
    return self.encode(input)
