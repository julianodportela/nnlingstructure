#!/bin/bash
#SBATCH --job-name=nllb_eval_ablation
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_ablation_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_ablation_%j.err

# Run FLORES-200 devtest + ergative eval for a specific checkpoint.
#
# Usage: sbatch jobs/eval_ablation.sh <checkpoint_path> <output_tag>
#   checkpoint_path  path to a saved model directory (or "baseline" to use NLLB-200)
#   output_tag       subdirectory name under outputs/ablations/

CHECKPOINT=${1:?"usage: sbatch eval_ablation.sh <checkpoint_path|baseline> <output_tag>"}
TAG=${2:?"usage: sbatch eval_ablation.sh <checkpoint_path|baseline> <output_tag>"}

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

if [ ! -f eval/ergative_test.json ]; then
    echo "[info] building ergative test set..."
    python src/build_ergative_testset.py \
        --data-dir $SCRATCH/nnling_data \
        --output   eval/ergative_test.json \
        --split    devtest
fi

CKPT_ARG=""
[ "$CHECKPOINT" != "baseline" ] && CKPT_ARG="--model-path $CHECKPOINT"

echo "=== FLORES-200 devtest ($TAG) ==="
python src/baseline_eval.py \
    --split devtest \
    --data-dir $SCRATCH/nnling_data \
    --output-dir outputs/ablations/${TAG} \
    $CKPT_ARG \
    --batch-size 8 --num-beams 4 --max-new-tokens 256

echo "=== Ergative-absolutive eval ($TAG) ==="
python src/eval_ergative.py \
    --test-file eval/ergative_test.json \
    --output-dir outputs/ablations/${TAG}_ergative \
    $CKPT_ARG \
    --num-beams 4 --batch-size 8

echo "EVAL_DONE=1"
