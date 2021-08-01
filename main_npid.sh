#!/bin/bash

source ~/lemniscate.pytorch/venv/bin/activate

python3 train_moco_ins.py \
 --batch_size 128 \
 --num_workers 24 \
 --nce_k 4096 \
 --dataset imagenet100 \
 --model resnet50 \
 --data_folder ~/IN100/Img/ \
 --model_path run3_npid/ \
 --tb_path run3_npid/tb


# python3 eval_moco_ins.py \
#  --model resnet50 \
#  --num_workers 24 \
#  --learning_rate 10 \
#  --data_folder ~/IN100/Img/ \
#  --model_path run2_npid/InsDis_nce_4096_resnet50_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ/ckpt_epoch_240.pth \
#  --tb_path run2_npid/tb3 \
#  --save_path run3_npid/lp_save \
#  --resume run2_npid/lp_save/InsDis_nce_4096_resnet50_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ_bsz_256_lr_10.0_decay_0_crop_0.2_aug_CJ/resnet50_layer6.pth

deactivate