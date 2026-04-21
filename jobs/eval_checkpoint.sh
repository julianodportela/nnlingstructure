#!/bin/bash
#SBATCH --job-name=nllb_eval
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_%j.err

# Evaluate any checkpoint against FLORES-200 devtest.
#
# Usage:
#   sbatch jobs/eval_checkpoint.sh <checkpoint_path> <output_tag>
#
# Examples:
#   sbatch jobs/eval_checkpoint.sh $SCRATCH/nllb_checkpoints/best finetuned_best
#   sbatch jobs/eval_checkpoint.sh $SCRATCH/nllb_checkpoints/epoch_2 epoch2

CHECKPOINT=${1:?"usage: sbatch eval_checkpoint.sh <checkpoint_path> <output_tag>"}
TAG=${2:-checkpoint}

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure
source .venv/bin/activate

python src/baseline_eval.py \
    --split       devtest \
    --batch-size  8 \
    --num-beams   4 \
    --max-new-tokens 256 \
    --data-dir    $SCRATCH/nnling_data \
    --output-dir  outputs/eval_$TAG \
    --model-path  $CHECKPOINT
