#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

user_dir=/workspace/AVTSR/avtnet_deltalm


python -B ${user_dir}/infer_st.py --config-dir ./conf/ --config-name st_decode.yaml \
  dataset.gen_subset=test common_eval.path=${user_dir}/433h_data_idea2/checkpoints/checkpoint_best.pt \
  common_eval.results_path=${user_dir}/generate \
  override.modalities=['audio','video']