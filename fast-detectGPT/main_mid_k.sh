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
res_path=$exp_path/results
mkdir -p $exp_path $data_path $res_path

# datasets="xsum squad writing" (ORIGINAL)
# Add parameter arrays
datasets="xsum"
source_models="gpt2"
mid_k_num_positions_values="1 5 10 20 30 50 60 70 80 90 100 150"    # Different number of positions to sample
mid_k_limit_values="2"            # Different vocabulary limits

# preparing dataset
for D in $datasets; do
  for M in $source_models; do
    for num_pos in $mid_k_num_positions_values; do
      for limit in $mid_k_limit_values; do
        echo `date`, Preparing dataset ${D}_${M}_pos${num_pos}_limit${limit} ...
        python scripts/data_builder.py --dataset $D --n_samples 500 --base_model_name $M \
                                       --output_file $data_path/${D}_${M}_pos${num_pos}_limit${limit} \
                                       --do_mid_k --mid_k_num_positions $num_pos --mid_k_limit $limit --mid_k_start_pos 5
      done
    done
  done
done

# White-box Setting
echo `date`, Evaluate models in the white-box setting :

# Evaluate Fast-DetectGPT and fast baselines
for D in $datasets; do
  for M in $source_models; do
    for num_pos in $mid_k_num_positions_values; do
      for limit in $mid_k_limit_values; do
        dataset_suffix="_pos${num_pos}_limit${limit}"
        
        echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}${dataset_suffix} ...
        python scripts/fast_detect_gpt.py --sampling_model_name $M --scoring_model_name $M --dataset $D \
                              --dataset_file $data_path/${D}_${M}${dataset_suffix} \
                              --output_file $res_path/${D}_${M}${dataset_suffix}

        echo `date`, Evaluating baseline methods on ${D}_${M}${dataset_suffix} ...
        python scripts/baselines.py --scoring_model_name $M --dataset $D \
                              --dataset_file $data_path/${D}_${M}${dataset_suffix} \
                              --output_file $res_path/${D}_${M}${dataset_suffix}
      done
    done
  done
done

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


# # Black-box Setting
# echo `date`, Evaluate models in the black-box setting:
# scoring_models="nvidia-9b"

# # evaluate Fast-DetectGPT
# for D in $datasets; do
#   for M in $source_models; do
#     M1=nvidia-9b # sampling model
#     for M2 in $scoring_models; do
#       echo `date`, Evaluating Fast-DetectGPT on ${D}_${M}.${M1}_${M2} ...
#       python scripts/fast_detect_gpt.py --sampling_model_name ${M1} --scoring_model_name ${M2} --dataset $D \
#                           --dataset_file $data_path/${D}_${M} --output_file $res_path/${D}_${M}.${M1}_${M2}
#     done
#   done
# done

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
