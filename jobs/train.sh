#!/bin/bash
#SBATCH --job-name=nllb_finetune
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/train_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/train_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure
source .venv/bin/activate

python src/train.py \
    --data-dir      $SCRATCH/nnling_data \
    --checkpoint-dir $SCRATCH/nllb_checkpoints \
    --output-dir    outputs/finetuned \
    --tatoeba-limit      100000 \
    --translate-weight   0.8 \
    --supertag-loss-weight 1.0 \
    --lr             2e-5 \
    --warmup-steps   500 \
    --batch-size     8 \
    --eval-batch-size 8 \
    --eval-num-beams 4 \
    --max-new-tokens 256 \
    --max-epochs     20 \
    --patience       3
