#!/bin/bash
python3.5 train.py \
  --gpus 0,1,2 \
  --n_workers 2 \
  --ckpt_dir $HOME/LOAE/DDPAE-video-prediction/ckpt \
  --dset_name moving_mnist \
  --evaluate_every 20 \
  --lr_init 1e-3 \
  --lr_decay 1 \
  --n_iters 50000 \
  --batch_size 64 \
  --n_components 2 \
  --stn_scale_prior 2 \
  --ckpt_name 50kTest
