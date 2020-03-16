import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
import scipy.io as sio
import pandas as pd


def load_data(root, is_train):


  if is_train:
    data_filename = 'train_pose_data.npy'
    dist_filename = 'train_pose_dist.npy'
    neigh_filename = 'train_pose_neigh.npy'

  else:
    data_filename = 'val_pose_data.npy'
    dist_filename = 'val_pose_dist.npy'
    neigh_filename = 'val_pose_neigh.npy'

  data_path = os.path.join(root, data_filename)
  dist_path = os.path.join(root, dist_filename)
  neigh_path = os.path.join(root, neigh_filename)
  inp = torch.FloatTensor(np.load(data_path))
  dist = torch.FloatTensor(np.load(dist_path))
  neigh = torch.LongTensor(np.load(neigh_path))

  return inp, neigh, dist


def main():
    X, y, neigh, dist = load_data('/data/BDSC/datasets/DFaust_Synthetic_Mocap', True)
    dset = data.TensorDataset(X, y, neigh, dist)
    for i, (x,y,n,d) in enumerate(data.DataLoader(dset, 2)):
      print(x, '\n', y, '\n', n, '\n', d, '\n')
      if i > 3:
        break

if __name__ == "__main__":
    main()