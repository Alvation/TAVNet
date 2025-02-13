#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1


lrs3_root=/workspace/AVTSR/data/lrs3
user_dir=/workspace/AVTSR/hubert
ckpt_name=hubert_ckpt
vocab_size=10000

fairseq-hydra-train --config-dir ${user_dir}/conf --config-name train_hubert.yaml \
  task.data=${user_dir}/30h_data task.label_dir=${user_dir}/30h_data \
  task.tokenizer_bpe_model=${lrs3_root}/spm${vocab_size}/spm_unigram${vocab_size}.model \
  model.w2v_path=${user_dir}/utils/hubert_base_ls960.pt \
  common.user_dir=${user_dir} hydra.run.dir=${user_dir}/${ckpt_name} \
  