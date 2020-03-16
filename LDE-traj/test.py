import glob
import numpy as np
import os
from PIL import Image
import torch
import data
import models
import utils
from utils.visualizer import Visualizer

# def save_images(prediction, gt, latent, save_dir, step):
#   pose, components = latent['pose'].data.cpu(), latent['components'].data.cpu()
#   batch_size, n_frames_total = prediction.shape[:2]
#   n_components = components.shape[2]
#   for i in range(batch_size):
#     filename = '{:05d}.png'.format(step)
#     y = gt[i, ...]
#     rows = [y]
#     if n_components > 1:
#       for j in range(n_components):
#         p = pose[i, :, j, :]
#         comp = components[i, :, j, ...]
#         if pose.size(-1) == 3:
#           comp = utils.draw_components(comp, p)
#         rows.append(utils.to_numpy(comp))
#     x = prediction[i, ...]
#     rows.append(x)
#     # Make a grid of 4 x n_frames_total images
#     image = np.concatenate(rows, axis=2).squeeze(1)
#     image = np.concatenate([image[i] for i in range(n_frames_total)], axis=1)
#     image = (image * 255).astype(np.uint8)
#     # Save image
#     Image.fromarray(image).save(os.path.join(save_dir, filename))
#     step += 1
#
#   return step

def evaluate(opt, dloader, model, use_saved_file=False):
  # Visualizer
  if hasattr(opt, 'save_visuals') and opt.save_visuals:
    vis = Visualizer(os.path.join(opt.ckpt_path, 'tb_test'))
  else:
    opt.save_visuals = False

  model.setup(is_train=False)
  metric = utils.Metrics()
  results = {}

  if hasattr(opt, 'save_all_results') and opt.save_all_results:
    save_dir = os.path.join(opt.ckpt_path, 'results')
    os.makedirs(save_dir, exist_ok=True)
  else:
    opt.save_all_results = False


  count = 0
  results = []
  for step, data in enumerate(dloader):
    input, output, neigh, dist, man = data
    res_dict = model.test(*data)
    results.append(res_dict)
    # if opt.save_all_results:
    #   gt = np.concatenate([input.numpy(), gt.numpy()], axis=1)
    #   prediction = utils.to_numpy(dec_output)
    #   count = save_images(prediction, gt, latent)

    if (step + 1) % opt.log_every == 0:
      print('{}/{}'.format(step + 1, len(dloader)))
      # if opt.save_visuals:
      #   vis.add_images(model.get_visuals(), step, prefix='test')


  final_results = {'Rank_G':0,'Trace_K':0,'Local_geom':0,'Total_loss':0}
  N = len(results)
  for item in results:
      final_results['Rank_G'] += item['Rank_G']/N
      final_results['Trace_K'] += item['Trace_K']/N
      final_results['Local_geom'] += item['Local_geom']/N
      final_results['Total_loss'] += item['Total_loss']/N

  # MSE
  # results.update(metric.get_scores())
  # TODO: metric.reset?

  return final_results

def main():
  opt, logger, vis = utils.build(is_train=False)
  #loader gives 1
  dloader = data.get_data_loader(opt)
  print('Val dataset: {}'.format(len(dloader.dataset)))
  model = models.get_model(opt)

  for epoch in opt.which_epochs:
    # Load checkpoint
    if epoch == -1:
      # Find the latest checkpoint
      checkpoints = glob.glob(os.path.join(opt.ckpt_path, 'net*.pth'))
      assert len(checkpoints) > 0
      epochs = [int(filename.split('_')[-1][:-4]) for filename in checkpoints]
      epoch = max(epochs)
    logger.print('Loading checkpoints from {}, epoch {}'.format(opt.ckpt_path, epoch))
    model.load(opt.ckpt_path, epoch)

    results = evaluate(opt, dloader, model)
    for metric in results:
      logger.print('{}: {}'.format(metric, results[metric]))

if __name__ == '__main__':
  main()
