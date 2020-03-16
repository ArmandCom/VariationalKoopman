import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.networks.encoder import ImageEncoder

class Encoder(nn.Module):
  '''
  The backbone model. CNN + 2D LSTM.
  Given an input video, output the mean and standard deviation of the pose
  vectors (initial pose + beta) of each component.
  '''
  # def __init__(self, n_components, n_frames_output, n_channels, image_size,
  #              image_latent_size, hidden_size, ngf, output_size, independent_components):
  def __init__(self, n_components, n_frames_output, n_channels, image_size,
               image_latent_size, hidden_size, ngf, output_size, independent_components):
    super(Encoder, self).__init__()

    n_layers = int(np.log2(image_size)) - 1
    self.image_encoder = ImageEncoder(n_channels, image_latent_size, ngf, n_layers)
    # Encoder
    self.encode_rnn = nn.LSTM(image_latent_size + hidden_size, hidden_size,
                              num_layers=1, batch_first=True)
    # if independent_components:
    #   predict_input_size = hidden_size
    # else:
    #   predict_input_size = hidden_size * 2
    # self.predict_rnn = nn.LSTM(predict_input_size, hidden_size, num_layers=1, batch_first=True)

    # Betad
    self.fc_layer = nn.Linear(hidden_size, output_size)
    self.bnorm = nn.BatchNorm1d(output_size, affine=False)

    # Initial pose
    # self.initial_pose_rnn = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
    # self.initial_pose_mu = nn.Linear(hidden_size, output_size)
    # self.initial_pose_sigma = nn.Linear(hidden_size, output_size)

    self.n_components = n_components
    self.n_frames_output = n_frames_output
    self.image_latent_size = image_latent_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.independent_components = independent_components

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

  def encode(self, input):
    '''
    :param input: [N, T, C, H, W] --> [64, 10, 1, 64, 64]
    :return:
    '''
    '''
    First part of the model.
    input: video of size (batch_size, n_frames_input, n_channels, H, W)
    Return initial pose and input betas.
    '''
    batch_size, n_frames_input, n_channels, H, W = input.size()
    # encode each frame
    input_reprs = self.image_encoder(input.view(-1, n_channels, H, W))
    input_reprs = input_reprs.view(batch_size, n_frames_input, -1)
    # Note: They convert each input frame in a 256 vector, so the output of the image encoder is [N,T,256]

    # Initial zero hidden states (as input to lstm)
    # prev_hidden = [Variable(torch.zeros(batch_size, 1, self.hidden_size).cuda())] * n_frames_input
    # encoder_outputs = [] # all components
    # hidden_states = []
    # first_hidden_states = []
    #
    # # TODO: try without this. Maybe also without FC.
    # for i in range(self.n_components): # Note: In case of mnist n_comp = 2
    #   frame_outputs = []
    #   hidden = None
    #   for j in range(n_frames_input):
    #     rnn_input = torch.cat([input_reprs[:, j:(j+1), :], prev_hidden[j]], dim=2)
    #     # Note: rnn_input - concat 256 encoded inp + 64 hidden states
    #     output, hidden = self.encode_rnn(rnn_input, hidden)
    #     if hidden[0].size(0) == 1:
    #       # TODO: check that this is well done
    #       h = hidden[0]
    #       c = hidden[1]
    #     else:
    #       h = torch.cat([hidden[0][0:1], hidden[0][1:]], dim=2)
    #       c = torch.cat([hidden[1][0:1], hidden[1][1:]], dim=2)
    #     prev_hidden[j] = h.view(batch_size, 1, -1)
    #     frame_outputs.append(output)
    #
    #   frame_outputs = torch.cat(frame_outputs, dim=1) # for 1 component
    #   encoder_outputs.append(frame_outputs)
    #
    #   hidden_states.append((h, c))
    #
    # # batch_size x n_frames_input x n_components x hidden_size
    # encoder_outputs = torch.stack(encoder_outputs, dim=2)
    # Note: encoder_outputs concatenates the 2 components [N,T,ncomp,64]
    #
    # input_z = self.fc_layer(input_reprs).view(-1, self.output_size)

    # Get initial pose
    # first_hidden_states = torch.cat(first_hidden_states, dim=1)
    # initial_mu, initial_sigma = self.get_initial_pose(first_hidden_states)

    return input_reprs, 0


  def forward(self, input):
    '''
    param input: video of size (batch_size, n_frames_input, n_channels, H, W)
    Output: input_beta: mean and std (for reconstruction), shape
                (batch_size, n_frames_input, n_components, output_size)
            pred_beta: mean and std (for prediction), shape
                (batch_size, n_frames_output, n_components, output_size)
            initial_pose: mean and std
    '''
    input_z, encoder_outputs = self.encode(input)
    # input_z = self.bnorm(input_z) # TODO: why?
    return input_z, encoder_outputs
