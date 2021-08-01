#!/bin/bash

source ~/lemniscate.pytorch/venv/bin/activate
# python3 train_CMC.py \
#  --model resnet50v1 \
#  --batch_size 128 \
#  --num_workers 24 \
#  --data_folder ~/IN100/Img/ \
#  --model_path run1/ \
#  --tb_path run1/tb

python3 LinearProbing.py --dataset imagenet100 \
 --data_folder ~/IN100/Img/ \
 --save_path run1/lp_save \
 --tb_path run1/tb2 \
 --model_path run1/memory_nce_16384_resnet50v1_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_240.pth \
 --model resnet50v1 --learning_rate 10 --layer 6 \
 --resume run1/lp_save/calibrated_memory_nce_16384_resnet50v1_lr_0.03_decay_0.0001_bsz_128_view_Lab_bsz_256_lr_10.0_decay_0_view_Lab/resnet50v1_layer6.pth

deactivate