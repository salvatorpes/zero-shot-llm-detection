#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=icl_create
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=output/slurm_output_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

# Your job starts in the directory where you call sbatch
cd $HOME/DL4NLP-Group15/fast-detectGPT

# run the setup script (uncomment this once)
bash setup.sh 

# activate your conda environment
source activate fastdetectGPT

# Login to Hugging Face if not already logged in
# Load .env file if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ ! -f "$HOME/.cache/huggingface/token" ]; then
    echo "Logging in to Hugging Face CLI..."
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        echo "Error: HUGGINGFACE_TOKEN not set in .env file"
        exit 1
    fi
    # $CONDA_PREFIX/bin/huggingface-cli login --token "$HUGGINGFACE_TOKEN"
    hf auth login --token "$HUGGINGFACE_TOKEN"
else
    echo "Already logged in to Hugging Face."
fi


# Run your code
bash main.sh