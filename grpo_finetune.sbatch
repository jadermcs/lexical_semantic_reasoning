#!/bin/bash
#SBATCH --job-name=qwen-wic-grpo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu           # adjust to your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1         # 1 GPU; bump to 2+ if OOM on 1.7B + G=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com   # optional: remove or replace

# ── environment ──────────────────────────────────────────────────────────────
module purge
# Uncomment/adjust the CUDA module name for your cluster, e.g.:
# module load cuda/12.4
# module load python/3.11

# Activate your virtual / conda environment
# conda activate <env>
# source .venv/bin/activate

# ── housekeeping ──────────────────────────────────────────────────────────────
mkdir -p logs

export WANDB_PROJECT="wic-grpo"
# Uncomment to disable wandb if you don't have it set up on the cluster:
# export WANDB_MODE=disabled

# HuggingFace cache — point to a fast scratch dir if available
# export HF_HOME=/scratch/$USER/.cache/huggingface

# ── run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

python src/grpo_finetune.py
