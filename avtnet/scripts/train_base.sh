#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1


lrs3_root=/workspace/AVTSR/data/lrs3
user_dir=/workspace/AVTSR/avtnet
ckpt_name=anet_ckpt
vocab_size=10000

fairseq-hydra-train --config-dir ${user_dir}/conf --config-name train_base.yaml \
  task.data=${lrs3_root}/433h_data task.label_dir=${lrs3_root}/433h_data \
  task.tokenizer_bpe_model=${lrs3_root}/spm${vocab_size}/spm_unigram${vocab_size}.model \
  common.user_dir=${user_dir} hydra.run.dir=${user_dir}/${ckpt_name} \
  common.tensorboard_logdir=${user_dir}/${ckpt_name}/tensorboard \
  