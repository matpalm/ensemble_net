#!/usr/bin/env bash
set -ex

# large config
# python3 train.py \
#     --num-models 6 \
#     --max-conv-size 96 \
#     --dense-kernel-size 96 \
#     --batch-size 32 \
#     --epochs 2

python3 train.py \
  --max-conv-size 16 \
  --dense-kernel-size 16 \
  --steps-per-batch 1
  --epochs 2 \
  --models-per-device 2

# python3 train.py \
#     --input-mode single \
#     --num-models 4 \
#     --max-conv-size 32 \
#     --dense-kernel-size 32 \
#     --batch-size 32 \
#     --epochs 2 \
#     --model-dropout
