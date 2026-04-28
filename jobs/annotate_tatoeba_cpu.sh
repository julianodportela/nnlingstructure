#!/bin/bash
#SBATCH --job-name=annotate_cpu
#SBATCH --partition=education
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/annotate_cpu_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/annotate_cpu_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
# stanza parallelism: give it all CPUs we allocated
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure

source .venv/bin/activate

python -c "import stanza" 2>/dev/null || pip install --quiet "stanza>=1.7"

python src/spinoff/annotate_tatoeba.py \
    --data-dir    $SCRATCH/nnling_data \
    --output      $SCRATCH/nnling_data/tatoeba_annotated.jsonl \
    --stanza-dir  $SCRATCH/stanza_resources \
    --limit       100000 \
    --fmt         supertag+deprel \
    --chunk-size  64 \
    --split       train || { echo "[ERROR] annotate_tatoeba.py failed"; exit 1; }

echo "[done] annotation complete"
