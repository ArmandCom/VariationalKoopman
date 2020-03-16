from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import scipy.io as sio

import data.video_transforms as vtransforms
from .moving_mnist import MovingMNIST
from .synth_trajectories_3d import *
from .small_norb import *
from .mocap import *


def get_data_loader(opt):
  if opt.dset_name == 'moving_mnist':
    transform = transforms.Compose([vtransforms.ToTensor()])
    dset = MovingMNIST(opt.dset_path, opt.is_train, opt.n_frames_input,
                       opt.n_frames_output, opt.num_objects, transform)

  elif opt.dset_name == 'traj_3d':

    X, y, neigh, dist, man = load_data(opt.dset_path, opt.is_train)
    dset = data.TensorDataset(X, y, neigh, dist, man)

  elif opt.dset_name == 'DFaust_Synthetic_Mocap':

    X, neigh, dist = load_data(opt.dset_path, opt.is_train)
    dset = data.TensorDataset(X, neigh, dist)

  #TODO: implement dataset from here

  elif opt.dset_name == 'small_norb':
     dat, neigh, dist = load_data_norb(opt.dset_path, opt.is_train)
     dset = data.TensorDataset(dat, neigh, dist)

  # elif opt.dset_name == 'small_norb':
  #   dset = SmallNORBDataset(dataset_root=opt.dset_path)

  else:
    raise NotImplementedError

  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader

