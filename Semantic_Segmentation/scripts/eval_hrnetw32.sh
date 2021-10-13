#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`
ckpt_path='./log/hrnetw32.pth'
config_path='baseline.hrnetw32'
python eval.py --ckpt_path=${ckpt_path} --config_path=${config_path}