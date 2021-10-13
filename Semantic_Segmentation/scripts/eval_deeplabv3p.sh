#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
export PYTHONPATH=$PYTHONPATH:`pwd`
ckpt_path='./log/deeplabv3p.pth'
config_path='baseline.deeplabv3p'
python eval.py --ckpt_path=${ckpt_path} --config_path=${config_path}