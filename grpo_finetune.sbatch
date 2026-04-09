#!/bin/bash
#SBATCH --job-name=qwen-wic-grpo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu           # adjust to your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com   # optional: remove or replace
#SBATCH --open-mode=append            # don't truncate logs on requeue

# ── environment ──────────────────────────────────────────────────────────────
module purge
# Uncomment/adjust the CUDA module name for your cluster, e.g.:
# module load cuda/12.4
# module load python/3.11

# Activate conda environment
# Replace CONDA_BASE with the output of: conda info | grep "base environment"
CONDA_BASE=/opt/miniconda3   # e.g. ~/miniconda3 or /opt/conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate <env>          # replace <env> with your environment name

# ── housekeeping ──────────────────────────────────────────────────────────────
mkdir -p logs

export WANDB_PROJECT="wic-grpo"
# Uncomment to disable wandb if you don't have it set up on the cluster:
# export WANDB_MODE=disabled

# HuggingFace cache — point to a fast scratch dir if available
# export HF_HOME=/scratch/$USER/.cache/huggingface

# ── run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

# Resubmit this job if it hits the time limit (SIGTERM sent 60s before kill)
resubmit() {
    echo "Time limit approaching — resubmitting job for continuation..."
    sbatch "$0"
}
trap resubmit SIGTERM

accelerate launch --num_processes 2 src/grpo_finetune.py &
wait $!
