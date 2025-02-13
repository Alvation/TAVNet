#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

user_dir=/workspace/AVTSR/avtnet_noise_large_transformer


python -B ${user_dir}/infer_avsr.py --config-dir ./conf/ --config-name s2s_decode.yaml \
  dataset.gen_subset=test common_eval.path=${user_dir}/noise_large_transformer/checkpoints/checkpoint_best.pt \
  common_eval.results_path=${user_dir}/generate \
  override.modalities=['audio','video']