#!/bin/bash

### Feature Table ###
# a9a 123
# ijcnn1 22
# covtype 54
# mnist28 752
# real-sim 20958
# higgs 28
dataset=higgs

BASEDIR=$(dirname "$0")
OUTDIR="${BASEDIR}/ckpt/"

if [ ! -d "${OUTDIR}" ]
then   
    echo "Output dir ${OUTDIR} does not exist, creating..."
    mkdir -p ${OUTDIR}
fi    

CUDA_VISIBLE_DEVICES=0 python -u main_cls_cv.py \
    --feat_d 28 \
    --hidden_d 16 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 .0e-3 \
    --num_nets 40 \
    --data ${dataset} \
    --tr ${BASEDIR}/../data/${dataset}.train \
    --te ${BASEDIR}/../data/${dataset}.test \
    --batch_size 2048 \
    --epochs_per_stage 1 \
    --correct_epoch 1 \
    --model_order second \
    --normalization True \
    --cv True \
    --sparse False \
    --out_f ${OUTDIR}/${dataset}_cls.pth \
    --cuda
