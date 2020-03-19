from collections import defaultdict
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI#, Trace_ELBO
from models.custom_loss import Loss, _get_Gram, _get_Kernel

from .base_model import BaseModel
# from models.networks.pose_rnn import PoseRNN
from models.networks.mapping_rnn import Encoder, Decoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils


class DVLDE(BaseModel):
  '''
  The DVLDE model.
  '''
  def __init__(self, opt):
    super(DVLDE, self).__init__()

    self.lam = opt.weight_dim
    self.gam = opt.weight_lin
    self.eps = opt.slack_iso # Slack variable for local geometry
    self.delta = 1e-4 # To ensure logdet stability

    # For fractional dyn_loss
    # self.length_G = opt.length_G

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
    self.trans_rnn_hidden_size = opt.trans_rnn_hidden_size
    self.manifold_size = opt.manifold_size

    self.ngf = opt.ngf

    # Training parameters
    if opt.is_train:
      self.lr_init = opt.lr_init
      self.lr_decay = opt.lr_decay
      self.when_to_predict_only = opt.when_to_predict_only

    # Networks
    self.setup_networks()

    # Initial pose prior
    self.y_prior_mu = Variable(torch.cuda.FloatTensor([0]*self.manifold_size))
    self.y_prior_sigma = Variable(torch.cuda.FloatTensor([1]*self.manifold_size))

    self.time_enc_prior_mu = Variable(torch.cuda.FloatTensor([0]*self.time_enc_size))
    self.time_enc_prior_sigma = Variable(torch.cuda.FloatTensor([1]*self.time_enc_size))

    self.feat_prior_mu = Variable(torch.zeros(self.n_frames_input, self.feat_latent_size).cuda())
    self.feat_prior_sigma = Variable(torch.ones(self.n_frames_input, self.feat_latent_size).cuda())

  def setup_networks(self):
    '''
    Networks for DVLDE.
    '''
    self.nets = {}
    # These will be registered in model() and guide() with pyro.module().
    self.model_modules = {}
    self.guide_modules = {}

    # Encoder Features
    encoder_model = Encoder(self.n_frames_input, self.n_frames_output, self.n_channels, self.image_size,
                                    self.feat_latent_size, self.time_enc_size, self.t_enc_rnn_hidden_size,
                                  self.trans_rnn_hidden_size, self.manifold_size, self.ngf)
    self.encoder_model = nn.DataParallel(encoder_model.cuda())
    self.nets['encoder_model'] = self.encoder_model
    self.guide_modules['encoder_model'] = self.encoder_model

    # Backbone, Mapping RNN
    mapping_model = Decoder(self.n_frames_input, self.n_frames_output,
                                    self.feat_latent_size, self.time_enc_size, self.t_enc_rnn_hidden_size,
                                    self.trans_rnn_hidden_size, self.manifold_size)
    self.mapping_model = nn.DataParallel(mapping_model.cuda())
    self.nets['mapping_model'] = self.mapping_model
    self.model_modules['mapping_model'] = self.mapping_model

  def setup_training(self):
    '''
    Setup Pyro SVI, optimizers.
    '''
    if not self.is_train:
      return

    self.pyro_optimizer = optim.Adam({'lr': self.lr_init})

    loss = Loss(self.lam, self.gam, self.eps, self.delta)
    self.svis = {'elbo': SVI(self.model, self.guide, self.pyro_optimizer, loss=loss)}

    # Separate pose_model parameters and other networks' parameters
    # TODO: aquí poden haverhi problemes, perque pyro optimizer potser es el que es fa servir per sampling
    params = []
    for name, net in self.nets.items():
      if name != 'mapping_model':
        params.append(net.parameters())
    self.optimizer = torch.optim.Adam(\
                     [{'params': self.mapping_model.parameters(), 'lr': self.lr_init},
                      {'params': itertools.chain(*params), 'lr': self.lr_init}
                     ], betas=(0.5, 0.999))

  def sample_latent_prior(self, input):
    '''
    Return latent variables: [pose, z], sampled from prior distribution.
    '''
    latent = defaultdict(lambda: None)

    batch_size = input.size(0)
    # y prior
    N = batch_size  #TODO: check it's repeating well

    feat_prior_mu = self.feat_prior_mu.repeat(N,1,1)
    feat_prior_sigma = self.feat_prior_sigma.repeat(N, 1,1)
    input_repr = self.pyro_sample('features', dist.Normal, feat_prior_mu, feat_prior_sigma, sample=True)

    y_prior_mu = self.y_prior_mu.repeat(N,1)
    y_prior_sigma = self.y_prior_sigma.repeat(N, 1)
    y0 = self.pyro_sample('y_0', dist.Normal, y_prior_mu, y_prior_sigma, sample=True)

    time_enc_prior_mu = self.time_enc_prior_mu.repeat(N,1)
    time_enc_prior_sigma = self.time_enc_prior_sigma.repeat(N, 1)
    time_enc = self.pyro_sample('time_enc', dist.Normal, time_enc_prior_mu, time_enc_prior_sigma, sample=True)

    latent.update({'features': input_repr, 'y_0': y0, 'time_enc': time_enc})

    return latent

  def get_neigh_dist(self, man, neigh):

    K = torch.bmm(man, man.permute(0, 2, 1))
    i_index = torch.arange(0, man.size(1)).repeat(man.size(0), 1, 1).permute(0, 2, 1).cuda()
    neigh = neigh.long()
    neigh = torch.cat((i_index, neigh), 2)
    i = neigh[:, :, 0:1].cuda()
    j = neigh[:, :, 1:].cuda()

    Kii = torch.gather(K, 2, i)
    Kij = torch.gather(K, 2, j)
    Kjj = [K[b, j[b, :], j[b, :]].unsqueeze(0) for b in range(man.size(0))]
    Kjj = torch.cat(Kjj, 0)
    dist = Kii + Kjj - 2 * Kij  # TODO: check why this is sometimes negative, substitute for a more appropriate fn.

    return dist

  def encode(self, input, sample=True):
    '''
    Find observables by encoding and sampling with mapping_model
    and prediction.
    param input: video of size (batch_size, n_frames_input, C, dimx, dimy).
    param sample: True if this is called by guide(), and sample with pyro.sample.
    Return latent: a dictionary {'manifold': man}
    '''
    input_repr, time_enc, y0 = self.encoder_model(input, sample)
    latent = defaultdict(lambda: None)
    latent.update({'y_0': y0, 'time_enc': time_enc})

    return input_repr, latent

  def decode(self, neigh, latent, batch_size):
    '''
    Decode the latent variables into components, and produce the final output.
    param latent: dictionary, return values from self.encode()
    Return values:
    knn_dist: Distances to all the neighbors in time
    '''
    input_repr = latent['features']
    time_enc = latent['time_enc']
    y0 = latent['y_0']
    man = self.mapping_model(input_repr, time_enc, y0)
    man_dists = self.get_neigh_dist(man, neigh)

    return man_dists, man

  def model(self, input, output, neigh, ori_dists):
    '''
    Likelihood model: sample from prior, then decode to video.
    param input: video of size (batch_size, self.n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    param neigh: K nearest neighbors in time of each datapoint TODO: find shape
    param ori_dists: Original distances to knn given by the dataset
    '''
    # Register networks
    for name, net in self.model_modules.items():
      pyro.module(name, net)

    # Define observation
    # observation = torch.cat([input, output], dim=1)
    observation = ori_dists

    # Sample from prior
    latent = self.sample_latent_prior(input)

    # Decode
    man_dists, man = self.decode(neigh, latent, input.size(0))
    decoded_output = man_dists.view(*observation.size())

    # pyro observe
    sd = Variable(0.3 * torch.ones(*decoded_output.size()).cuda())
    pyro.sample('obs', dist.Normal(decoded_output, sd), obs=observation)

  def guide(self, input, output, neigh, ori_dists):
    '''
    Posterior model: encode input
    param input: video of size (batch_size, n_frames_input, C, H, W).
    parma output: not used.
    '''
    # Register networks
    for name, net in self.guide_modules.items():
      pyro.module(name, net)

    self.encode(input, sample=True)

  def train(self, input, output, neigh, ori_dists):
    '''
    param input: video of size (batch_size, n_frames_input, C, H, W)
    param output: video of size (batch_size, self.n_frames_output, C, H, W)
    Return video_dict, loss_dict
    '''
    input = Variable(input.cuda(), requires_grad=False).unsqueeze(-2).unsqueeze(-2)
    output = Variable(output.cuda(), requires_grad=False)
    neigh = Variable(neigh.cuda(), requires_grad=False)
    ori_dists = Variable(ori_dists.cuda(), requires_grad=False)
    assert input.size(1) == self.n_frames_input

    # SVI
    batch_size, _, C, H, W = input.size()
    numel = batch_size * self.n_frames_input * C * H * W
    # TODO: define numel for Dyn and Dim losses. It might be equal batch_size * self.n_frames_total
    numellatent = (self.n_frames_input+1 //2)
    loss_dict = {}
    for name, svi in self.svis.items():
      elbo, dyn_loss, dim_loss, man = svi.loss_and_grads(svi.model, svi.guide, input, output, neigh, ori_dists)
      loss_dict['elbo'] = elbo / numel
      loss_dict['dyn_loss'] = dyn_loss / numellatent
      loss_dict['dim_loss'] = dim_loss / numellatent
      loss_dict['Norm_of_M'] = torch.norm(man).item()

    # Update parameters
    self.optimizer.step()
    self.optimizer.zero_grad()

    return {}, loss_dict

  def test(self, input, output, neigh, ori_dists, epoch, save_every=1):
    '''
    Return decoded output.
    '''
    input = Variable(input.cuda()).unsqueeze(-2).unsqueeze(-2)
    output = Variable(output.cuda())
    neigh = Variable(neigh.cuda())
    ori_dists = Variable(ori_dists.cuda())

    batch_size, _, _, _, W = input.size()
    output = Variable(output.cuda())

    # gt = torch.cat([input, output], dim=1)
    gt = ori_dists

    res_dict = {}
    latent = self.encode(input, sample=False)
    man_dists, man = self.decode(neigh, latent, input.size(0))
    decoded_output = man_dists.view(*gt.size())


    # svG = np.linalg.svd(utils.to_numpy(_get_Gram(man)[0]))[1]
    # svG = svG/np.sum(svG)
    # svK = np.linalg.svd(utils.to_numpy(_get_Kernel(man)[0]))[1]
    # svK = svK/np.sum(svK)
    # res_dict['sv3_G_test'] = svG[2]
    # res_dict['sv3_K_test'] = svK[2]
    # res_dict['sv2_K_test'] = svK[1]
    if epoch % save_every:
      res_dict['plot'] = self.save_visuals(gt, man, None, epoch)

    self.save_visuals(gt, decoded_output, input, latent)
    return decoded_output.cpu(), latent

  def save_visuals(self, input, output, output_gt, epoch):

    M_plot = utils.to_numpy(output)
    inp_plot = utils.to_numpy(input)
    # M_plot = output.detach().cpu().numpy()
    # inp_plot = input.detach().cpu().numpy()

    if output_gt is not None:
      M_gt_plot = output_gt.numpy()

    plt.close()

    elev = 60
    angle = 45

    fig = plt.figure()
    ax1 = fig.add_subplot(231, projection='3d')
    # ax1.set_xlim(-5, 5)
    # ax1.set_ylim(-5, 5)
    #ax1.set_zlim(0, 16)
    ax1.view_init(elev=45, azim=0)
    ax1.plot(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2])
    ax1.scatter(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2],
                c=np.linspace(0, 1, M_plot.shape[1]))

    ax2 = fig.add_subplot(232, projection='3d')
    # ax2.set_xlim(-5, 5)
    # ax2.set_ylim(-5, 5)
    #ax1.set_zlim(0, 16)
    ax2.view_init(elev=-45, azim=0)
    ax2.plot(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2])
    ax2.scatter(M_plot[0,:,0],M_plot[0,:,1], M_plot[0,:,2],
                c=np.linspace(0, 1, M_plot.shape[1]))

    ax3 = fig.add_subplot(233)
    # ax3.set_xlim(-2, 2)
    # ax3.set_ylim(-2, 2)
    ax3.plot(M_plot[0,:,0],M_plot[0,:,1],)
    ax3.scatter(M_plot[0,:,0],M_plot[0,:,1],
                c=np.linspace(0, 1, inp_plot.shape[1]))

    ax4 = fig.add_subplot(234, projection='3d')
    ax4.view_init(elev=elev, azim=angle)
    # ax4.set_xlim(-5, 5)
    # ax4.set_ylim(-5, 5)
    ax4.set_zlim(0, 16)

    ax4.plot(inp_plot[0, :, 0],inp_plot[0, :, 1],inp_plot[0, :, 2])
    ax4.scatter(inp_plot[0, :, 0],inp_plot[0, :, 1],inp_plot[0, :, 2],
                c=np.linspace(0, 1, inp_plot.shape[1]))

    ax5 = fig.add_subplot(235)
    # ax5.set_xlim(-2, 2)
    # ax5.set_ylim(-2, 2)
    if output_gt is not None:
      ax5.plot(M_gt_plot[0, :, 0], M_gt_plot[0, :, 1])
      ax5.scatter(M_gt_plot[0, :, 0],M_gt_plot[0, :, 1],
                  c=np.linspace(0, 1, inp_plot.shape[1]))
    # fig.savefig('test_ep' + str(epoch) + '.png')
    return fig

  def update_hyperparameters(self, epoch, n_epochs):
    '''
    If when_to_predict_only > 0 and it halfway through training, then only train with
    prediction loss.
    '''
    lr_dict = super(DVLDE, self).update_hyperparameters(epoch, n_epochs)

    if self.when_to_predict_only > 0 and epoch > int(n_epochs * self.when_to_predict_only):
      self.predict_loss_only = True

    return lr_dict

  # def save_visuals(self, gt, output, components, latent):
  #   '''
  #   Save results. Draw bounding boxes on each component.
  #   '''
  #   super(DVLDE, self).save_visuals(gt, output, components, latent)