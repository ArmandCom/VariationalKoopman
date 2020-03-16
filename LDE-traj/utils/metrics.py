import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils

class Metrics(object):
  '''
  Evaluation metric: BCE and MSE.
  '''
  def __init__(self):
    self.mse_loss = nn.MSELoss()
    self.mse_rec_results = []
    self.mse_pred_results = []
    self.last_sv = []

  def update(self, gt_inp, rec):
    """
    gt, pred are tensors of size (..., 1, H, W) in the range [0, 1].
    """
    C, H, W = gt_inp.size()[-3:]
    if isinstance(gt_inp, torch.Tensor):
      gt = Variable(gt_inp)
    if isinstance(rec, torch.Tensor):
      rec = Variable(rec)

    mse_rec_score = self.mse_loss(rec, gt_inp)
    eps = 1e-4
    rec.data[rec.data < eps] = eps
    rec.data[rec.data > 1 - eps] = 1 -eps

    mse_rec_score = mse_rec_score.item()
    self.mse_rec_results.append(mse_rec_score)

  def get_scores(self):
    mse_rec_score = np.mean(self.mse_rec_results)
    scores = {'mse rec': mse_rec_score}
    return scores

  def reset(self):
    self.mse_rec_results = []