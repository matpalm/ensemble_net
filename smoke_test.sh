#!/usr/bin/env bash
set -ex

python3 train.py \
    --num-models 6 \
    --max-conv-size 96 \
    --dense-kernel-size 96 \
    --batch-size 32 \
    --epochs 2
