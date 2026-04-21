#!/bin/bash
#SBATCH --job-name=nllb_baseline
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/baseline_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/baseline_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache

# Clear PYTHONPATH so the Lmod Python bundle doesn't shadow venv packages.
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure

source .venv/bin/activate

python src/baseline_eval.py \
    --split devtest \
    --batch-size 8 \
    --num-beams 4 \
    --max-new-tokens 256 \
    --data-dir $SCRATCH/nnling_data
