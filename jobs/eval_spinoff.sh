#!/bin/bash
#SBATCH --job-name=eval_spinoff
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_spinoff_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/eval_spinoff_%j.err

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
    print('[ERROR] CUDA not available', flush=True)
    sys.exit(1)
print(f'[cuda] OK  device={torch.cuda.get_device_name(0)}', flush=True)
"

# Find the most recent spinoff best checkpoint.
SPINOFF_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
    | grep spinoff | sort | tail -1)

if [ -z "$SPINOFF_CKPT" ]; then
    echo "[ERROR] no spinoff checkpoint found under $SCRATCH/nllb_checkpoints/spinoff_*"
    exit 1
fi
echo "[info] spinoff checkpoint: $SPINOFF_CKPT"

# Build the ergative test set if not already present.
if [ ! -f eval/ergative_test.json ]; then
    echo "[info] building ergative test set..."
    python src/build_ergative_testset.py \
        --data-dir $SCRATCH/nnling_data \
        --output   eval/ergative_test.json \
        --split    devtest
fi

# ── Ergative eval on the spinoff checkpoint ───────────────────────────────
echo ""
echo "=== Ergative-absolutive eval — spinoff ==="
python src/eval_ergative.py \
    --model-path $SPINOFF_CKPT \
    --test-file  eval/ergative_test.json \
    --output-dir outputs/spinoff_eval/ergative \
    --num-beams  4 \
    --batch-size 8

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " SPINOFF vs BASELINE SUMMARY"
echo "════════════════════════════════════════════════════════════════"

python - <<'EOF'
import json, pathlib

def load(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else None

baseline_dev  = load("outputs/baseline/metrics.devtest.json")
spinoff_dev   = load("outputs/spinoff/metrics.devtest.json")
baseline_erg  = load("outputs/full_eval/baseline_ergative/metrics.json")
spinoff_erg   = load("outputs/spinoff_eval/ergative/metrics.json")

def row(label, dev, erg):
    if dev:
        print(f"  [{label}] BLEU={dev['bleu']:.2f}  chrF={dev['chrf']:.2f}")
    if erg:
        print(f"  [{label}] ERG  hit={erg['ergative_hit_rate']:.1%}  err={erg['ergative_error_rate']:.1%}  "
              f"ABS hit={erg['absolutive_hit_rate']:.1%}  err={erg['absolutive_error_rate']:.1%}")

print()
print("Baseline NLLB-200 (unmodified):")
row("baseline", baseline_dev, baseline_erg)
print()
print("Spinoff (stanza-annotated Tatoeba MTL):")
row("spinoff ", spinoff_dev, spinoff_erg)

if baseline_dev and spinoff_dev:
    delta_bleu = spinoff_dev['bleu'] - baseline_dev['bleu']
    delta_chrf = spinoff_dev['chrf'] - baseline_dev['chrf']
    print()
    print(f"  delta BLEU={delta_bleu:+.2f}  delta chrF={delta_chrf:+.2f}")

if baseline_erg and spinoff_erg:
    delta_erg  = spinoff_erg['ergative_hit_rate']  - baseline_erg['ergative_hit_rate']
    delta_aerr = spinoff_erg['ergative_error_rate'] - baseline_erg['ergative_error_rate']
    print(f"  delta ERG hit={delta_erg:+.1%}  delta ERG err={delta_aerr:+.1%}")
EOF

echo ""
echo "[info] full outputs → outputs/spinoff_eval/"
