#!/bin/bash
#SBATCH --job-name=nllb_trans_only
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/train_translation_only_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/train_translation_only_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
unset PYTHONPATH

cd /home/ling3850_jdp69/nnlingstructure

module load CUDA/12.8.0
source .venv/bin/activate

python -c "import torch; assert 'cu128' in torch.__version__" 2>/dev/null || \
    pip install --quiet --force-reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128

python -c "
import torch, sys
if not torch.cuda.is_available():
    print(f'[ERROR] CUDA not available (torch {torch.__version__})', flush=True)
    sys.exit(1)
print(f'[cuda] OK  device={torch.cuda.get_device_name(0)}  torch={torch.__version__}', flush=True)
"

RUN_DIR=$SCRATCH/nllb_checkpoints/trans_only_$(date +%Y%m%d_%H%M%S)
echo "[info] checkpoint dir: $RUN_DIR"

python src/train.py \
    --data-dir      $SCRATCH/nnling_data \
    --checkpoint-dir $RUN_DIR \
    --output-dir    outputs/ablations/translation_only \
    --tatoeba-limit      100000 \
    --translation-only \
    --lr             2e-5 \
    --warmup-steps   500 \
    --batch-size     8 \
    --eval-batch-size 8 \
    --eval-num-beams 4 \
    --max-new-tokens 256 \
    --max-epochs     20 \
    --patience       999

echo "[info] best checkpoint: $RUN_DIR/best"
echo "CKPT_PATH=$RUN_DIR/best"
