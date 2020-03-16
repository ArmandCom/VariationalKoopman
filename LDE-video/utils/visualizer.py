import numpy as np
import os
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import scipy
import matplotlib as plt
from .misc import *
import torch

class Visualizer:
  def __init__(self, tb_path):
    self.tb_path = tb_path

    if os.path.exists(tb_path):
      # if prompt_yes_no('{} already exists. Proceed?'.format(tb_path)):
      os.system('rm -r {}'.format(tb_path))
      # else:
      #   exit(0)

    self.writer = SummaryWriter(tb_path)
    self.savedir = '/data/BDSC/results/'
    self.eval_every = 20

  def add_scalar(self, scalar_dict, epoch, global_step=None):
    for tag, scalar in scalar_dict.items():
      if isinstance(scalar, dict):
        self.writer.add_scalars(tag, scalar, epoch)
      elif isinstance(scalar, plt.figure.Figure):
        self.writer.add_figure(tag, scalar, epoch)
      elif tag == 'Embedding' or tag == 'Original-Domain':
            # labels = np.linspace(0, scalar.shape[0], scalar.shape[0])
            # labels = np.expand_dims(np.arange(scalar.shape[0]), axis=1)
            # labels = np.expand_dims(labels, axis=1)
            # labels = torch.tensor(np.expand_dims(labels, axis=1))
            self.writer.add_embedding(
              scalar,
                tag = tag,
                global_step=global_step)
      elif isinstance(scalar, list) or isinstance(scalar, np.ndarray):
        continue
      else:
        self.writer.add_scalar(tag, scalar, epoch)

  def add_images(self, image_dict, epoch, global_step=None, prefix=None):
    for tag, images in image_dict.items():
      if prefix is not None:
        tag = '{}/{}'.format(prefix, tag)
      images = torch.clamp(images, -1, 1)
      images = vutils.make_grid(images, nrow=images.size(0), normalize=True, range=(-1, 1))

      '''Save images of results'''
      # if epoch % self.eval_every == 0 and epoch != 0:
      #   case = self.tb_path.split('/')[-2]
      #   resImageDir = os.path.join(self.savedir, 'figures', case)
      #   if not os.path.exists(resImageDir):
      #     os.makedirs(resImageDir)
      #   scipy.misc.imsave(os.path.join(resImageDir, prefix + '_step-' + str(global_step).zfill(5) + '_epoch-' + str(epoch).zfill(3) + '.png'), images[:, :130].permute(1,2,0))

      self.writer.add_image(tag, images, global_step)
