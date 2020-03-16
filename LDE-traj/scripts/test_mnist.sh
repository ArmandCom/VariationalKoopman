#!/bin/bash
python test.py \
  --gpus 0 \
  --n_workers 4 \
  --batch_size 16 \
  --dset_name moving_mnist \
  --ckpt_dir $HOME/DDPAE-video-prediction/ckpt \
  --log_every 5 \
  --save_visuals 0 \
  --save_results 1 \
  --ckpt_name crop_NC2_lr1.0e-03_bt64_200k


python test.py \
  --gpus 0 --n_workers 4 --batch_size 16 --dset_name moving_mnist --ckpt_dir /home/armandcomas/LOAE/MNIST/th_rankdef_v2/ckpt --log_every 5 --save_visuals 0 --save_results 1
  --ckpt_name crop_NC1_lr5.0e-04_bt96_lam0001_rec-pred_128bn