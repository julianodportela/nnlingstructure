#!/bin/bash
#SBATCH --job-name=annotate_tatoeba
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/annotate_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/annotate_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure

module load CUDA/12.8.0

source .venv/bin/activate

# Install stanza if not present.
python -c "import stanza" 2>/dev/null || pip install --quiet "stanza>=1.7"

python src/spinoff/annotate_tatoeba.py \
    --data-dir    $SCRATCH/nnling_data \
    --output      $SCRATCH/nnling_data/tatoeba_annotated.jsonl \
    --stanza-dir  $SCRATCH/stanza_resources \
    --limit       100000 \
    --fmt         supertag+deprel \
    --chunk-size  128 \
    --split       train || { echo "[ERROR] annotate_tatoeba.py failed"; exit 1; }

echo "[done] annotation complete"
