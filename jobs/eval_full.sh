#!/bin/bash
#SBATCH --job-name=nllb_eval_full
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_full_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_full_%j.err

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

# Build the ergative test set from UD + FLORES-200 if not already present.
if [ ! -f eval/ergative_test.json ]; then
    echo "[info] building ergative test set..."
    python src/build_ergative_testset.py \
        --data-dir $SCRATCH/nnling_data \
        --output   eval/ergative_test.json \
        --split    devtest
fi

# Find the most recent fine-tuned best checkpoint, if one exists.
# run_* dirs are timestamped so alphabetical sort = chronological.
BEST_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
    | sort | tail -1)

# ── 1. Full FLORES-200 devtest — baseline ──────────────────────────────────
echo ""
echo "=== [1/4] Full devtest BLEU/chrF — baseline ==="
python src/baseline_eval.py \
    --split      devtest \
    --data-dir   $SCRATCH/nnling_data \
    --output-dir outputs/full_eval/baseline \
    --batch-size 8 \
    --num-beams  4 \
    --max-new-tokens 256

# ── 2. Full FLORES-200 devtest — fine-tuned ────────────────────────────────
if [ -n "$BEST_CKPT" ]; then
    echo ""
    echo "=== [2/4] Full devtest BLEU/chrF — fine-tuned ($BEST_CKPT) ==="
    python src/baseline_eval.py \
        --split      devtest \
        --data-dir   $SCRATCH/nnling_data \
        --output-dir outputs/full_eval/finetuned \
        --model-path $BEST_CKPT \
        --batch-size 8 \
        --num-beams  4 \
        --max-new-tokens 256
else
    echo ""
    echo "=== [2/4] skipped — no fine-tuned checkpoint found ==="
fi

# ── 3. Ergative-absolutive eval — baseline ────────────────────────────────
echo ""
echo "=== [3/4] Ergative-absolutive alignment — baseline ==="
python src/eval_ergative.py \
    --test-file  eval/ergative_test.json \
    --output-dir outputs/full_eval/baseline_ergative \
    --num-beams  4 \
    --batch-size 8

# ── 4. Ergative-absolutive eval — fine-tuned ──────────────────────────────
if [ -n "$BEST_CKPT" ]; then
    echo ""
    echo "=== [4/4] Ergative-absolutive alignment — fine-tuned ($BEST_CKPT) ==="
    python src/eval_ergative.py \
        --model-path $BEST_CKPT \
        --test-file  eval/ergative_test.json \
        --output-dir outputs/full_eval/finetuned_ergative \
        --num-beams  4 \
        --batch-size 8
else
    echo ""
    echo "=== [4/4] skipped — no fine-tuned checkpoint found ==="
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " SUMMARY"
echo "════════════════════════════════════════════════"

for label in baseline finetuned; do
    mfile="outputs/full_eval/${label}/metrics.devtest.json"
    if [ -f "$mfile" ]; then
        bleu=$(python -c "import json; d=json.load(open('$mfile')); print(f\"{d['bleu']:.2f}\")")
        chrf=$(python -c "import json; d=json.load(open('$mfile')); print(f\"{d['chrf']:.2f}\")")
        echo "[$label] full devtest  BLEU=$bleu  chrF=$chrf"
    fi
    efile="outputs/full_eval/${label}_ergative/metrics.json"
    if [ -f "$efile" ]; then
        python -c "
import json
d = json.load(open('$efile'))
erg  = d.get('ergative_hit_rate')
aerr = d.get('ergative_error_rate')
abs_ = d.get('absolutive_hit_rate')
berr = d.get('absolutive_error_rate')
print(f'[$label] ergative subset  BLEU={d[\"bleu\"]:.2f}  chrF={d[\"chrf\"]:.2f}')
if erg  is not None: print(f'  ERG hit={erg:.1%}  error={aerr:.1%}')
if abs_ is not None: print(f'  ABS hit={abs_:.1%}  error={berr:.1%}')
"
    fi
done

echo ""
echo "[info] detailed outputs → outputs/full_eval/"
