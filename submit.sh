#!/bin/bash


GPU=$1
CONFIG=$2
SUBMUT=$3

CUDA_VISIBLE_DEVICES=${GPU} python3 ./src/submit.py \
    --config "${CONFIG}" \
    --submit-path "${SUBMUT}"
