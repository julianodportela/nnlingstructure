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

# Load CUDA 12.8 system libraries (driver version 12080 = CUDA 12.8 max).
module load CUDA/12.8.0

source .venv/bin/activate

# If the wrong torch build is in the venv (e.g. after a fresh pip install -r), fix it.
python -c "import torch; assert 'cu128' in torch.__version__" 2>/dev/null || \
    pip install --quiet --force-reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128

# Fast-fail: abort immediately rather than training silently on CPU for 23 hours.
python -c "
import torch, sys
if not torch.cuda.is_available():
    print(f'[ERROR] CUDA not available (torch {torch.__version__}, cuda={torch.version.cuda})', flush=True)
    sys.exit(1)
print(f'[cuda] OK  device={torch.cuda.get_device_name(0)}  torch={torch.__version__}', flush=True)
"

RUN_DIR=$SCRATCH/nllb_checkpoints/run_$(date +%Y%m%d_%H%M%S)
echo "[info] checkpoint dir: $RUN_DIR"

python src/train.py \
    --data-dir      $SCRATCH/nnling_data \
    --checkpoint-dir $RUN_DIR \
    --output-dir    outputs/finetuned \
    --tatoeba-limit      100000 \
    --translate-weight   0.65 \
    --supertag-loss-weight 1.0 \
    --lr             2e-5 \
    --warmup-steps   500 \
    --batch-size     8 \
    --eval-batch-size 8 \
    --eval-num-beams 4 \
    --max-new-tokens 256 \
    --max-epochs     20 \
    --patience       5
