#!/bin/bash

source ~/lemniscate.pytorch/venv/bin/activate

IS_TRAIN=0
IS_EVAL=1
IS_EXTRACT=0

if [ ${IS_TRAIN} -eq 1 ]; then
    python3 mask-npid/train_mask_ins.py \
    --batch_size 128 \
    --num_workers 20 \
    --softmax \
    --learning_rate 0.03 \
    --mask_learning_rate 0.003 \
    --nce_k 16384 \
    --nce_t 0.07 \
    --nce_m 0.5 \
    --dataset imagenet100 \
    --model resnet50 \
    --data_folder ~/IN100/Img/ \
    --model_path run3_mask_npid_bn_sigm/ \
    --print_freq 200 \
    --tb_path run3_mask_npid_bn_sigm/tb
fi

if [ ${IS_EVAL} -eq 1 ]; then
    python3 mask-npid/eval_mask_ins.py \
    --model resnet50 \
    --num_workers 24 \
    --learning_rate 30 \
    --layer 6 \
    --extract ${IS_EXTRACT} \
    --data_folder ~/IN100/Img/ \
    --model_path run3_mask_npid_bn_sigm/InsDis_softmax_16384_resnet50_lr_0.03_masklr_0.003_decay_0.0001_bsz_128_crop_0.2_aug_CJ/ckpt_epoch_240.pth \
    --save_path run3_mask_npid_bn_sigm/lp_save \
    --tb_path run3_mask_npid_bn_sigm/lp_save/tb \
    --resume run3_mask_npid_bn_sigm/lp_save/InsDis_softmax_16384_resnet50_lr_0.03_masklr_0.003_decay_0.0001_bsz_128_crop_0.2_aug_CJ/resnet50_layer6.pth
fi

deactivate