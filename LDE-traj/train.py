import copy
import numpy as np
import os

import data
import models
import utils
import sys
from test import evaluate

opt, logger, vis = utils.build(is_train=True, tb_dir='tb_train')
train_loader = data.get_data_loader(opt)
# Validation set
val_opt = copy.deepcopy(opt)
val_opt.is_train = False
val_opt.num_objects = [1] # Only matters for MNIST
val_loader = data.get_data_loader(val_opt)
print('Val dataset: {}'.format(len(val_loader.dataset)))

model = models.get_model(opt)

save_every = 5

if opt.load_ckpt_dir != '':
  ckpt_dir = os.path.join(opt.ckpt_dir, opt.dset_name, opt.load_ckpt_dir)
  assert os.path.exists(ckpt_dir)
  logger.print('Loading checkpoint from {}'.format(ckpt_dir))
  model.load(ckpt_dir, opt.load_ckpt_epoch)

opt.n_epochs = max(opt.n_epochs, opt.n_iters // len(train_loader))
logger.print('Total epochs: {}'.format(opt.n_epochs))
mode = opt.ckpt_path.split('/')[-1]
save_latent_every = 20 # TODO: check sincro with eval_every in visualizer.py
savedir = '/data/BDSC/results/'
for epoch in range(opt.start_epoch, opt.n_epochs):
  model.setup(is_train=True)
  print('Train epoch', epoch)
  hp_dict = model.update_hyperparameters(epoch, opt.n_epochs)
  vis.add_scalar(hp_dict, epoch)

  # ph_old = None
  for step, data in enumerate(train_loader):
    input, output, neigh, dists, man = data
    # _, loss_dict, ph_old = model.train(*data, ph_old)
    loss_dict = model.train(*data)
    if step % opt.log_every == 0:
      # Write to tensorboard
      if step !=0:
        vis.add_scalar(loss_dict, epoch, epoch * len(train_loader) + step)
      # Visualization
      res_dict = model.test(*data, epoch=epoch, save_every=save_every)
      # vis.add_images(model.get_visuals(), epoch, epoch * len(train_loader) + step, prefix='train')
      # Random sample test data
      idx = np.random.randint(len(val_loader.dataset))
      inputs = val_loader.dataset[idx:idx+1]
      res_dict = model.test(*inputs, epoch=1, save_every=save_every)
      if step !=0:
        vis.add_scalar(res_dict, epoch, epoch * len(train_loader) + step)
      # vis.add_images(model.get_visuals(), epoch, epoch * len(train_loader) + step, prefix='test')

  logger.print('Epoch {}/{}:{}'.format(epoch, opt.n_epochs-1, mode))
  if epoch > 400:
    break

  # Evaluate on val set
  if opt.evaluate_every > 0 and (epoch + 1) % opt.evaluate_every == 0 and \
      opt.n_frames_output > 0:
    results = evaluate(val_opt, val_loader, model)
    vis.add_scalar(results, epoch)
    for metric in results.keys():
      logger.print('{}: {}'.format(metric, results[metric]))

  # Save model checkpoints
  if (epoch + 1) % opt.save_every == 0 and epoch > 0 or epoch == opt.n_epochs - 1:
    model.save(opt.ckpt_path, epoch + 1)