#!/bin/bash

GPU=$1
TTA=$2
WORK_DIR=~/kaggle/gen-eng/cnn1d_smooth/b32_Adam_lr0.001_f0/best.pth
CUDA_VISIBLE_DEVICES=${GPU} python3 ./src/predict.py \
    --load ${WORK_DIR} \
    --tta ${TTA}
