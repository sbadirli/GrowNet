#!/bin/bash

### Feature Table ###
# a9a 123
# ijcnn1 22
# covtype 54
# mnist28 752
# real-sim 20958
# criteo 45
# Criteo_Dracula 766
# yahoo.pair 519
# higgs 28
data=higgs

CUDA_VISIBLE_DEVICES=0 python main_cls_cv_experiments.py \
    --feat_d 28 \
    --hidden_d 32 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 .0e-3 \
    --num_nets 40 \
    --data ${data} \
    --tr /home/sbadirli/GBNN/data/${data}.train \
    --te /home/sbadirli/GBNN/data/${data}.test \
    --batch_size 2048 \
    --epochs_per_stage 1 \
    --correct_epoch 1 \
    --model_order second \
    --normalization True \
    --cv True \
    --sparse False \
    --out_f ./ckpt/${data}_cls.pth \
    --cuda
