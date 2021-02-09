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
  --max-conv-size 8 \
  --dense-kernel-size 8 \
  --steps-per-batch 2 \
  --epochs 2 \
  --models-per-device 2 \
  --sample-data \
  --log-level DEBUG


# python3 train.py \
#     --input-mode single \
#     --num-models 4 \
#     --max-conv-size 32 \
#     --dense-kernel-size 32 \
#     --batch-size 32 \
#     --epochs 2 \
#     --model-dropout
