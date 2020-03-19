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


def load_mnist(root):
  # Load MNIST dataset for generating training data.
  path = os.path.join(root, 'train-images-idx3-ubyte.gz')
  with gzip.open(path, 'rb') as f:
    mnist = np.frombuffer(f.read(), np.uint8, offset=16)
    mnist = mnist.reshape(-1, 28, 28)
  return mnist


def load_data(root, is_train):

  if is_train:
    filename = 'Project_Dataset_train.mat'
    path = os.path.join(root, filename)
    dataset = sio.loadmat(path)
    data = dataset[list(dataset.keys())[-1]]
  else:
    filename = 'Project_Dataset_val.mat'
    # if flag == 'testing':
    #   filename = 'Project_Dataset_test.mat'
    path = os.path.join(root, filename)
    dataset = sio.loadmat(path)
    data = dataset[list(dataset.keys())[-1]]

  inp = torch.FloatTensor(np.stack(data['original_data'][0]))
  out = torch.FloatTensor(np.stack(data['manifold_data'][0]))
  neigh = torch.LongTensor(np.stack(data['connections'][0]))
  dist = torch.FloatTensor(np.stack(data['distances'][0]))
  # man = torch.FloatTensor(np.stack(data['manifold_noisy_data'][0]))
  man = torch.FloatTensor(np.stack(data['manifold_data'][0]))

  return inp, out, neigh, dist, man

def load_fixed_set(root, is_train):
  # Load the fixed dataset
  filename = 'mnist_test_seq.npy'
  path = os.path.join(root, filename)
  dataset = np.load(path)
  dataset = dataset[..., np.newaxis]
  return dataset

# class SynthTrajectories(data.Dataset):
#   def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
#                transform=None):
#
#     super(SynthTrajectories, self).__init__()
#
#     self.dataset = load_data(root, is_train)
#     self.length = self.dataset.shape[0] # shape dim=1?
#     self.is_train = is_train
#     self.n_frames_input = n_frames_input
#     self.n_frames_output = n_frames_output
#     self.n_frames_total = self.n_frames_input + self.n_frames_output
#     self.transform = transform # Note: needed?
#     # For generating data
#
#   def __getitem__(self, idx):
#
#     inp = self.dataset[0][idx]
#     out = self.dataset[1][idx]
#     neigh = self.dataset[2][idx]
#     dist = self.dataset[3][idx]
#
#     return inp, out, neigh, dist
#
#   def __len__(self):
#     return self.length # Is this length of the batch?

def main():
    X, y, neigh, dist = load_data('/data/BDSC/datasets/', True)
    dset = data.TensorDataset(X, y, neigh, dist)
    for i, (x,y,n,d) in enumerate(data.DataLoader(dset, 2)):
      print(x, '\n', y, '\n', n, '\n', d, '\n')
      if i > 3:
        break

if __name__ == "__main__":
    main()