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
res_path=$exp_path/results-blackbox
mkdir -p $exp_path $data_path $res_path

# datasets="xsum squad writing" (ORIGINAL)
datasets="xsum hc3" # A BASELINE + OUR DATASET
# source_models="gpt2-xl opt-2.7b gpt-neo-2.7B gpt-j-6B gpt-neox-20b" (ORIGINAL)
# source_models="gpt-oss-20b" (DOESN"T WORK -> TOO LARGE)
# source_models="nvidia-9b" # (WORKS WITH XSUM BUT NOT HC3)
source_models="r1-8b phi-2 mistral-7b" # OUR FUNCTIONING MODELS

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    echo `date`, Preparing dataset ${D}_${M} ...
    python scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M --output_file $data_path/${D}_${M}
  done
done

# White-box Setting
echo `date`, Evaluate models in the white-box setting:

# # Evaluate Fast-DetectGPT, fast baselines and DetectGPT (but doesn't work)
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating Fast-DetectGPT on ${D}_${M} ...
#     python scripts/fast_detect_gpt.py --sampling_model_name $M --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

#     echo `date`, Evaluating baseline methods on ${D}_${M} ...
#     python scripts/baselines.py --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}

#     echo `date`, Evaluating DetectGPT on ${D}_${M} ...
#     python scripts/detect_gpt.py --mask_filling_model_name ${M} --scoring_model_name ${M} --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}_slow
#   done
# done

# evaluate DNA-GPT
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DNA-GPT on ${D}_${M} ...
#     python scripts/dna_gpt.py --base_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
#   done
# done

# evaluate DetectGPT and its improvement DetectLLM
# for D in $datasets; do
#   for M in $source_models; do
#     echo `date`, Evaluating DetectGPT on ${D}_${M} ...
#     python scripts/detect_gpt.py --scoring_model_name $M --mask_filling_model_name t5-3b --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}
#      # we leverage DetectGPT to generate the perturbations
#     echo `date`, Evaluating DetectLLM methods on ${D}_${M} ...
#     python scripts/detect_llm.py --scoring_model_name $M --dataset $D \
#                           --dataset_file $data_path/${D}_${M}.t5-3b.perturbation_100 --output_file $res_path/${D}_${M}
#   done
# done


# Black-box Setting
echo `date`, Evaluate models in the black-box setting:
scoring_models="mistral-7b"

# evaluate Fast-DetectGPT
for D in $datasets; do
  for M in $source_models; do
    M1="phi-2" # sampling model
    for M2 in $scoring_models; do
      echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
      python scripts/fast_detect_gpt.py --sampling_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
                          --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
    done
  done
done

# evaluate DetectGPT and its improvement DetectLLM
# for D in $datasets; do
#   for M in $source_models; do
#     M1=t5-3b  # perturbation model
#     for M2 in $scoring_models; do
#       echo `date`, Evaluating DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python scripts/detect_gpt.py --mask_filling_model_name ${M1} --scoring_model_name ${M2} --n_perturbations 100 --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
#       # we leverage DetectGPT to generate the perturbations
#       echo `date`, Evaluating DetectLLM methods on ${D}_${M}.${M1}_${M2} ...
#       python scripts/detect_llm.py --scoring_model_name ${M2} --dataset $D \
#                           --dataset_file $data_path/${D}_${M}.${M1}.perturbation_100 --output_file $res_path/${D}_${M}.${M1}_${M2}
#     done
#   done
# done

# delete pycache
find . -name "__pycache__" -type d -exec rm -rf {} +