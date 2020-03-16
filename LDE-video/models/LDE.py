from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .base_model import BaseModel
from models.networks.mapping import Encoder
from models.networks.inv_mapping import Decoder
# from models.networks.encoder import ImageEncoder
# from models.networks.decoder import ImageDecoder
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# torch.backends.cudnn.enabled = False
# TODO: trick! Switch from run to debug and vice versa --> hold Shift.

class LDE(BaseModel):
  '''
  The DDPAE model.
  '''
  def __init__(self, opt):
    super(LDE, self).__init__()

    self.alpha = opt.weight_dim
    self.beta = opt.weight_local_geom
    self.gamma = opt.weight_lin
    self.pie = opt.weight_rec

    # self.length_G = opt.length_G

    self.eps = opt.slack_iso
    self.delta = 1e-2

    self.is_train = opt.is_train
    self.image_size = opt.image_size[-1]

    # Data parameters
    # self.__dict__.update(opt.__dict__)
    self.n_channels = opt.n_channels
    self.batch_size = opt.batch_size
    self.n_frames_input = opt.n_frames_input
    self.n_frames_output = opt.n_frames_output
    self.n_frames_total = self.n_frames_input + self.n_frames_output

    # Dimensions
    self.feat_latent_size = opt.feat_latent_size
    self.time_enc_size = opt.time_enc_size
    self.t_enc_rnn_hidden_size = opt.t_enc_rnn_hidden_size
    self.t_enc_rnn_output_size = opt.t_enc_rnn_output_size
    self.manifold_size = opt.manifold_size

    self.ngf = opt.ngf
    self.predict_loss_only = False

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay
      self.when_to_predict_only = opt.when_to_predict_only
      self.swap_lin_loss = opt.swap_lin_loss
      self.alternating_loss = opt.alternating_loss

    # Networks
    self.setup_networks()

    # Losses
    self.lin_trace = True
    lin_mode = 'ld'
    if lin_mode == 'ld':
        self.loss_lin = lambda M: torch.logdet(get_G(M) + self.delta * torch.eye(get_G(M).shape[-1])
                                                  .repeat(M.size(0), 1, 1).cuda()).mean()
        # self.loss_lin = lambda M: torch.logdet(get_partial_G(M, L=self.length_G) + self.delta * torch.eye(self.length_G)
        #                                           .repeat(M.size(0)*(M.size(1) - 2*self.length_G + 1), 1, 1).cuda()).mean()
    elif lin_mode == 'tr':
        self.loss_lin = lambda M, flag='g': get_trace_K(M, flag).mean()

    self.loss_lin_tr = lambda M, flag='g': get_trace_K(M, flag).mean()

    self.loss_mse = nn.MSELoss() # reduction 'mean'
    self.loss_l1 = nn.L1Loss(reduction='none')

    self.loss_dim = lambda M, flag='k': -get_trace_K(M, flag).mean()
    self.loss_dist = lambda M, neigh, ori_dist: (self.loss_l1(get_dist(M, neigh),(ori_dist**2).cuda())/(ori_dist.cuda()+self.eps)).mean()# - self.eps * ori_dist**2

  def setup_networks(self):
    '''
    Networks for DDPAE.
    '''
    self.nets = {}
    self.deepest_shape = (64, self.image_size // 4, self.image_size // 4)  # 64, 24, 24

    # Mapping
    # TODO: find pretrained model
    encoder_model = Encoder(self.n_frames_input, self.n_frames_output, self.n_channels,
                         self.image_size, self.feat_latent_size, self.time_enc_size,
                         self.ngf, self.t_enc_rnn_output_size, self.t_enc_rnn_hidden_size, self.manifold_size)
    self.encoder_model = nn.DataParallel(encoder_model.cuda())
    self.nets['encoder_model'] = self.encoder_model


    # Inverse Mapping

    # n_layers = int(np.log2(self.image_size)) - 1
    # decoder_model = Decoder(self.manifold_size, self.n_channels, self.ngf, n_layers)

    decoder_model = Decoder(
      code_length=self.manifold_size,
      deepest_shape=self.deepest_shape,
      output_shape=[self.n_channels, self.image_size, self.image_size]
    )

    self.encoder_model = nn.DataParallel(encoder_model.cuda())
    self.decoder_model = nn.DataParallel(decoder_model.cuda())
    self.nets['decoder_model'] = self.decoder_model

    # Image encoder and decoder
    # n_layers = int(np.log2(self.object_size)) - 1
    # object_encoder = ImageEncoder(self.n_channels, self.content_latent_size,
    #                               self.ngf, n_layers)
    # object_decoder = ImageDecoder(self.manifold_size, self.n_channels,
    #                               self.ngf, n_layers, 'sigmoid') # default content_latent_size
    # self.object_encoder = nn.DataParallel(object_encoder.cuda())
    # self.object_decoder = nn.DataParallel(object_decoder.cuda())
    # self.nets.update({'object_encoder': self.object_encoder,
    #                   'object_decoder': self.object_decoder})
    # self.model_modules['decoder'] = self.object_decoder
    # self.guide_modules['encoder'] = self.object_encoder

  def setup_training(self):
    '''
    Setup optimizers.
    '''
    if not self.is_train:
      return

    params = []
    # for name, net in self.nets.items():
    #   # if name != 'encoder_model': # Note: Impose decoder different optimizer than encoder
    #   params.append(net.parameters())
    self.optimizer = torch.optim.Adam(\
                     [{'params': self.encoder_model.parameters(), 'lr': self.lr_init},
                      {'params': self.decoder_model.parameters(), 'lr': self.lr_init}
                     ], betas=(0.9, 0.999))

  def encode(self, input):
    return self.encoder_model(input)

  def decode(self, latent):
    return self.decoder_model(latent)

  def train(self, input, neigh, ori_dist, step):
    '''
    param input: video of size (batch_size, n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    Return video_dict, loss_dict
    '''
    input = Variable(input.cuda(), requires_grad=False)
    # output = Variable(output.cuda(), requires_grad=False)
    # assert input.size(1) == self.n_frames_input

    if len(input.shape) == 4:
      input = input.unsqueeze(2)
    batch_size, n_frames_input, chan, n_dimx, n_dimy = input.size()
    # gt = torch.cat([input, output], dim=1)

    numel = batch_size * n_frames_input * n_dimx * n_dimy
    loss_dict = {}

    '''Encode'''
    encoder_input = input
    M = self.encode(encoder_input)
    meanM = M.mean(dim=1, keepdims=True)
    M = M - meanM

    '''Decode'''
    decoded_output = self.decode(M + meanM).view(batch_size, n_frames_input, chan, n_dimx, n_dimy)
    decoded_output = decoded_output.clamp(0, 1)#.squeeze()

    '''Losses'''
    # loss_mse_rec = self.loss_mse(gt[:, :self.n_frames_input],
    #                              decoded_output[:, :self.n_frames_input])
    # loss_dict['MSE_rec'] = loss_mse_rec


    '''Linearity loss'''
    if self.alternating_loss:
      if step % self.swap_lin_loss:
        last_svG = np.linalg.svd(get_G(M)[0].data.cpu().numpy())[1][-1].mean()
        # print(last_svG)
        self.lin_trace = last_svG > 1e-7

      if self.lin_trace:
        loss_lin = self.loss_lin_tr(M)
      else:
        loss_lin = self.loss_lin(M)

    else:
      # loss_lin = self.loss_lin(M)
      loss_lin = 0

    # loss_dict['Rank_G'] = loss_lin.item()

    '''Dimensionality loss'''
    # loss_dim = self.loss_dim(M)
    loss_dim = 0
    # loss_dict['Trace_K'] = loss_dim.item() #Note: remove bias - standarize?

    '''Local Geometry loss'''
    # loss_dist = self.loss_dist(M, neigh, ori_dist)
    loss_dist = 0
    # loss_dict['Local_geom'] = loss_dist.item()
    '''Reconstruction Loss'''
    loss_rec = self.loss_mse(decoded_output, input)
    # loss_rec = 0
    loss_dict['Reconstruction'] = loss_rec.item()

    '''Data norms'''
    loss_dict['Norm_of_M'] = torch.norm(M + meanM).item()
    # loss_dict['Norm_of_inp'] = torch.norm(input).item()
    # loss_dict['Norm_of_out'] = torch.norm(decoded_output).item()
    # loss_dict['Rate_of_norms'] = (torch.norm(input)/torch.norm(decoded_output)).item()


    '''Optimizer step'''
    loss = self.gamma * loss_lin + self.alpha * loss_dim + self.beta * loss_dist + self.pie * loss_rec
    loss_dict['Total_loss'] = loss.item()
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss_dict

  def test(self, input, neigh, ori_dist, epoch=0, save_every=1):
    '''
    Return decoded output.
    '''
    res_dict = {}

    input = Variable(input.cuda())

    if len(input.shape) ==4:
      input = input.unsqueeze(2)

    batch_size, n_frames_input, chan, n_dimx, n_dimy = input.size()

    # gt = torch.cat([input, output], dim=1)
    gt = input

    '''Encode'''
    encoder_input = input
    M = self.encode(encoder_input)
    meanM = M.mean(dim=1, keepdims=True)
    M = M - meanM

    '''Decode'''
    decoded_output = self.decode(M + meanM).view(batch_size, n_frames_input, 1, n_dimx, n_dimy)
    decoded_output = decoded_output.clamp(0, 1)#.squeeze(2)

    # '''Losses'''
    # # *Fidelity*
    # # loss_mse_rec = self.loss_mse(gt[:, :self.n_frames_input],
    # #                              decoded_output[:, :self.n_frames_input])
    # # res_dict['MSE_rec_test'] = loss_mse_rec
    # # res_dict['MSE_pred_test'] = loss_mse_pred

    # loss_lin = self.loss_lin(M)
    # res_dict['zTest_Rank_G'] = loss_lin.item()
    #
    # loss_dim = self.loss_dim(M)
    # res_dict['zTest_Trace_K'] = loss_dim.item()
    #
    # loss_dist = self.loss_dist(M, neigh, ori_dist)
    # res_dict['zTest_Local_geom'] = loss_dist.item()

    loss_rec = self.loss_mse(decoded_output, input)
    res_dict['zTest_Reconstruction'] = loss_rec.item()

    # svG = np.linalg.svd(get_G(M)[0].data.cpu().numpy())[1]
    # svG = svG/np.sum(svG)
    # svK = np.linalg.svd(get_K(M)[0].data.cpu().numpy())[1]
    # svK = svK/np.sum(svK)

    if epoch % save_every:
      # res_dict['Plot'] = self.save_visuals(svK, svG,  epoch)
      self.save_visuals_img(gt, decoded_output)

      # res_dict['Embedding'] = M[0].data
      # res_dict['Original-Domain'] = input[0].data

    return res_dict

  def save_visuals(self, svk, svg, epoch):
    '''
    Save results. Draw bounding boxes on each component.
    '''
    # svk = svk.detach().cpu().numpy()
    # svg = svk.detach().cpu().numpy()

    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    svk_perc = []
    for i in range(svk.shape[0]):
      svk_perc.append(np.array([svk[i]]*int(np.round(svk[i]*1000))))

    svk = np.concatenate(svk_perc, axis=0)
    im1 = ax1.imshow(np.expand_dims(svk, 0).repeat(100,0))
    ax1.set_title('SV of K')

    ax2 = fig.add_subplot(212)
    svg_perc = []
    for i in range(svg.shape[0]):
      svg_perc.append([svg[i]] * int(np.round(svg[i]*1000)))

    svg = np.concatenate(svg_perc, axis=0)
    im2 = ax2.imshow(np.expand_dims(svg, 0).repeat(100,0))
    ax2.set_title('SV of G')

    # fig.savefig('test_ep' + str(epoch) + '.png')
    return fig

  def save_visuals_img(self, gt, output):
    '''
    Save results. Draw bounding boxes on each component.
    '''

    super(LDE, self).save_visuals_img(gt, output)

    # def save_visuals(self, input, output, epoch):
    #   '''
    #   Save results. Draw bounding boxes on each component.
    #   '''
    #   M_plot = output.detach().cpu().numpy()
    #   inp_plot = input.detach().cpu().numpy()
    #
    #   plt.close()
    #
    #   elev = 60
    #   angle = 45
    #
    #   fig = plt.figure()
    #   ax1 = fig.add_subplot(231, projection='3d')
    #   # ax1.set_xlim(-5, 5)
    #   # ax1.set_ylim(-5, 5)
    #   # ax1.set_zlim(0, 16)
    #   ax1.view_init(elev=45, azim=0)
    #   ax1.plot(M_plot[0, :, 0], M_plot[0, :, 1], M_plot[0, :, 2])
    #   ax1.scatter(M_plot[0, :, 0], M_plot[0, :, 1], M_plot[0, :, 2],
    #               c=np.linspace(0, 1, M_plot.shape[1]))
    #
    #   ax2 = fig.add_subplot(232, projection='3d')
    #   # ax2.set_xlim(-5, 5)
    #   # ax2.set_ylim(-5, 5)
    #   # ax1.set_zlim(0, 16)
    #   ax2.view_init(elev=-45, azim=0)
    #   ax2.plot(M_plot[0, :, 0], M_plot[0, :, 1], M_plot[0, :, 2])
    #   ax2.scatter(M_plot[0, :, 0], M_plot[0, :, 1], M_plot[0, :, 2],
    #               c=np.linspace(0, 1, M_plot.shape[1]))
    #
    #   ax3 = fig.add_subplot(233)
    #   # ax3.set_xlim(-2, 2)
    #   # ax3.set_ylim(-2, 2)
    #   ax3.plot(M_plot[0, :, 0], M_plot[0, :, 1], )
    #   ax3.scatter(M_plot[0, :, 0], M_plot[0, :, 1],
    #               c=np.linspace(0, 1, inp_plot.shape[1]))
    #
    #   ax4 = fig.add_subplot(234, projection='3d')
    #   ax4.view_init(elev=elev, azim=angle)
    #   # ax4.set_xlim(-5, 5)
    #   # ax4.set_ylim(-5, 5)
    #   ax4.set_zlim(0, 16)
    #
    #   ax4.plot(inp_plot[0, :, 0], inp_plot[0, :, 1], inp_plot[0, :, 2])
    #   ax4.scatter(inp_plot[0, :, 0], inp_plot[0, :, 1], inp_plot[0, :, 2],
    #               c=np.linspace(0, 1, inp_plot.shape[1]))
    #
    #   # fig.savefig('test_ep' + str(epoch) + '.png')
    #   return fig

    # super(LDE, self).save_visuals(gt, output, latent)

  def update_hyperparameters(self, epoch, n_epochs):
    '''
    If when_to_predict_only > 0 and it halfway through training, then only train with
    prediction loss.
    '''
    lr_dict = super(LDE, self).update_hyperparameters(epoch, n_epochs)

    if self.when_to_predict_only > 0 and epoch > int(n_epochs * self.when_to_predict_only):
      self.predict_loss_only = True

    return lr_dict


# def normalize(x, eps=1e-5):
#
#     # seq_norm = nn.BatchNorm1d(x.size(1), affine=False).cuda(gpu_id)
#     mean = torch.mean(x, 1).unsqueeze(1)
#     std = x.std(1).unsqueeze(1) + eps
#
#     x = (x - mean).div(std)
#
#     return x, mean, std
#
# def denormalize(x, mean, std):
#
#     x = (x * std) + mean
#
#     return x