#!/bin/bash
#SBATCH --job-name=nllb_ergeval
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/ergeval_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/ergeval_%j.err

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

# Build the test set from UD + FLORES-200 if not already present.
if [ ! -f eval/ergative_test.json ]; then
    echo "[info] building ergative test set..."
    python src/build_ergative_testset.py \
        --data-dir $SCRATCH/nnling_data \
        --output   eval/ergative_test.json \
        --split    devtest
fi

# Evaluate baseline NLLB-200.
echo "[info] evaluating baseline..."
python src/eval_ergative.py \
    --test-file  eval/ergative_test.json \
    --output-dir outputs/ergative_eval/baseline \
    --num-beams  4 \
    --batch-size 8

# Evaluate the best fine-tuned checkpoint if one exists.
BEST_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
    | sort | tail -1)

if [ -n "$BEST_CKPT" ]; then
    echo "[info] evaluating fine-tuned checkpoint: $BEST_CKPT"
    python src/eval_ergative.py \
        --model-path $BEST_CKPT \
        --test-file  eval/ergative_test.json \
        --output-dir outputs/ergative_eval/finetuned \
        --num-beams  4 \
        --batch-size 8
else
    echo "[warn] no fine-tuned checkpoint found — skipping finetuned eval"
fi
