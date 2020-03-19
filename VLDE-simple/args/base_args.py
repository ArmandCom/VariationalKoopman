import argparse
import os


class BaseArgs:
  '''
  Arguments for data, model, and checkpoints.
  '''
  def __init__(self):
    self.is_train, self.split = None, None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
    self.parser.add_argument('--gpus', type=str, default='0', help='visible GPU ids, separated by comma')

    # data
    # Option 1: Moving MNIST / MD-Mov-MNIST
    # self.parser.add_argument('--dset_dir', type=str, default=os.path.join(os.environ['HOME'], 'FactorDynamics/DDPAE-video-prediction'))
    # self.parser.add_argument('--dset_name', type=str, default='moving_mnist')
    # self.parser.add_argument('--image_size', type=int, nargs='+', default=[64, 64])
    # self.parser.add_argument('--num_objects', type=int, nargs='+', default=[1],
    #                          help='Max number of digits in Moving MNIST videos.')
    # self.parser.add_argument('--n_components', type=int, default=1)
    # Option 2: Trajectories 3D
    self.parser.add_argument('--dset_dir', type=str, default=os.path.join('/data/BDSC/', 'datasets/'))
    self.parser.add_argument('--dset_name', type=str, default='traj_3d')
    self.parser.add_argument('--image_size', type=int, nargs='+', default=[1, 3])

    # model
    self.parser.add_argument('--model', type=str, default='basic', help='Model name')
    self.parser.add_argument('--ngf', type=int, default=8,
                             help='number of channels in encoder and decoder')

    # dimensions
    self.parser.add_argument('--feat_latent_size', type=int, default=3,
                             help='Size of convolutional features')
    self.parser.add_argument('--time_enc_size', type=int, default=4,
                             help='Size of temporal encoding')
    self.parser.add_argument('--t_enc_rnn_hidden_size', type=int, default=8,
                             help='Size of the hidden size of the time enc rnn')
    self.parser.add_argument('--trans_rnn_hidden_size', type=int, default=8,
                             help='Size of the output size of the transition rnn')
    self.parser.add_argument('--manifold_size', type=int, default=3,
                             help='Dimension of the manifold for the given time sequence')

    # Changing hyperparameters
    # TODO: review n_frames_input
    self.parser.add_argument('--n_frames_input', type=int, default=49)
    self.parser.add_argument('--n_frames_output', type=int, default=0)

    self.parser.add_argument('--weight_dim', type=float, default=0.001,
                             help='Weight of the manifold dimension loss - alpha')
    self.parser.add_argument('--weight_local_geom', type=float, default=60000,
                             help='Weight of local geometry loss - beta')
    self.parser.add_argument('--weight_lin', type=float, default=35,
                             help='Weight of the linearity loss - gamma')
    self.parser.add_argument('--length_G', type=float, default=12,
                             help='Row/Column dimension of the generated Gram matrix')

    self.parser.add_argument('--slack_iso', type=float, default=0.0001,
                             help='Local geometry relaxation')

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.environ['HOME'], '/data/Armand/VLDE/tensorboard', 'ckpt'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt_name', type=str, default='ckpt', help='checkpoint name')
    self.parser.add_argument('--log_every', type=int, default=30, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=5, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1, help='evaluate on val set every x epochs')

  def parse(self):
    opt = self.parser.parse_args()

    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)
    if opt.is_train:
      ckpt_name = 'bt{:d}_{:s}'.format(
        opt.batch_size, opt.ckpt_name)
    else:
      ckpt_name = opt.ckpt_name
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, ckpt_name)

    # Hard code
    if opt.dset_name == 'moving_mnist':
      opt.n_channels = 1
      opt.image_size = (64, 64)
    elif opt.dset_name == 'traj_3d':
      opt.n_channels = 1
      opt.image_size = (1, 3)
    else:
      raise NotImplementedError

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
