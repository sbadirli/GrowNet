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
# ca_housing 8
# YearPredictionMSD 90
# slice_localization 384
data=YearPredictionMSD

CUDA_VISIBLE_DEVICES=1 python main_reg_cv.py \
    --feat_d 90 \
    --hidden_d 32 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 .0e-3 \
    --num_nets 40 \
    --data ${data} \
    --tr /var/opt/data/user_data/s.badirli/${data}_tr.npz \
    --te /var/opt/data/user_data/s.badirli/${data}_te.npz \
    --batch_size 2048 \
    --epochs_per_stage 1 \
    --correct_epoch 1 \
    --normalization True \
    --cv True \
    --out_f ./ckpt/${data}_cls.pth \
    --cuda
