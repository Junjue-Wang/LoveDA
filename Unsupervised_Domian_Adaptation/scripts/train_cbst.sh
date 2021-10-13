#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`
config_path='st.cbst.2urban'
python CBST_train.py --config_path=${config_path}
