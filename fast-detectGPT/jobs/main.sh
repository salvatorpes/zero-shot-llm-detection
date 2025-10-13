#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# setup the environment
echo `date`, Setup the environment ...
export TOKENIZERS_PARALLELISM=false
# set -e  # exit if error

# prepare folders
exp_path=exp_main
data_path=$exp_path/data
res_path=$exp_path/results-whitebox-api
mkdir -p $exp_path $data_path $res_path

datasets="xsum hc3" # A BASELINE + OUR DATASET
source_models="gpt-4o-mini" # OUR FUNCTIONING MODELS

# # preparing dataset
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Preparing dataset ${D}_${M} ...
#     python scripts/data_builder.py --dataset $D --n_samples 500 --do_top_k --base_model_name $M --output_file $data_path/${D}_${M}
#   done
# done

# White-box Setting
echo `date`, Evaluate models in the white-box setting:

# Evaluate Fast-DetectGPT, fast baselines and DetectGPT (but doesn't work)
for D in $datasets; do
  for M in $source_models; do
    # echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
    # python scripts/fast_detect_gpt.py --sampling_model_name $M --scoring_model_name $M --dataset $D \
    #                       --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_FD

    echo `date`, Evaluating baseline methods on ${D}_${M} ...
    python scripts/baselines.py --scoring_model_name $M --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_BL

  done
done

# delete pycache
find . -name "__pycache__" -type d -exec rm -rf {} +