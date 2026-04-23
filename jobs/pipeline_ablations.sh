#!/bin/bash
# Submit translation-only ablation and weight-sweep jobs in parallel,
# then run eval for each after training finishes.
#SBATCH --job-name=pipeline_ablations
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:59:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_ablations_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_ablations_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

wait_for_job() {
    local jid=$1
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
}

get_best_ckpt() {
    # Extract CKPT_PATH written by the train script to its log.
    local log=$1
    grep "^CKPT_PATH=" "$log" 2>/dev/null | tail -1 | cut -d= -f2-
}

cd /home/ling3850_jdp69/nnlingstructure

# ── Submit both training jobs ───────────────────────────────────────────────
TRANS_JID=$(sbatch jobs/train_translation_only.sh | awk '{print $4}')
echo "[ablations] translation-only training: job $TRANS_JID"

SWEEP_JID=$(sbatch jobs/train_weight_sweep.sh | awk '{print $4}')
echo "[ablations] weight-sweep training (tw=0.5): job $SWEEP_JID"

# ── Wait for translation-only, then submit eval ────────────────────────────
echo "[ablations] waiting for translation-only job $TRANS_JID..."
wait_for_job "$TRANS_JID"
echo "[ablations] translation-only training done"

TRANS_LOG="jobs/logs/train_translation_only_${TRANS_JID}.out"
TRANS_CKPT=$(get_best_ckpt "$TRANS_LOG")
if [ -z "$TRANS_CKPT" ]; then
    TRANS_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
        | grep trans_only | sort | tail -1)
fi
echo "[ablations] translation-only checkpoint: $TRANS_CKPT"

if [ -n "$TRANS_CKPT" ] && [ -d "$TRANS_CKPT" ]; then
    EVAL_TRANS_JID=$(sbatch jobs/eval_ablation.sh "$TRANS_CKPT" translation_only | awk '{print $4}')
    echo "[ablations] translation-only eval: job $EVAL_TRANS_JID"
else
    echo "[ablations] ERROR: no translation-only checkpoint found — skipping eval"
    EVAL_TRANS_JID=""
fi

# ── Wait for weight sweep, then submit eval ────────────────────────────────
echo "[ablations] waiting for weight-sweep job $SWEEP_JID..."
wait_for_job "$SWEEP_JID"
echo "[ablations] weight-sweep training done"

SWEEP_LOG="jobs/logs/train_weight_sweep_${SWEEP_JID}.out"
SWEEP_CKPT=$(get_best_ckpt "$SWEEP_LOG")
if [ -z "$SWEEP_CKPT" ]; then
    SWEEP_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
        | grep weight50 | sort | tail -1)
fi
echo "[ablations] weight-sweep checkpoint: $SWEEP_CKPT"

if [ -n "$SWEEP_CKPT" ] && [ -d "$SWEEP_CKPT" ]; then
    EVAL_SWEEP_JID=$(sbatch jobs/eval_ablation.sh "$SWEEP_CKPT" weight_sweep_050 | awk '{print $4}')
    echo "[ablations] weight-sweep eval: job $EVAL_SWEEP_JID"
else
    echo "[ablations] ERROR: no weight-sweep checkpoint found — skipping eval"
    EVAL_SWEEP_JID=""
fi

# ── Wait for both evals ────────────────────────────────────────────────────
[ -n "$EVAL_TRANS_JID" ] && { echo "[ablations] waiting for eval job $EVAL_TRANS_JID..."; wait_for_job "$EVAL_TRANS_JID"; }
[ -n "$EVAL_SWEEP_JID" ] && { echo "[ablations] waiting for eval job $EVAL_SWEEP_JID..."; wait_for_job "$EVAL_SWEEP_JID"; }

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " ABLATION SUMMARY"
echo "════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib

results = {
    "MTL run2 (tw=0.65)":  ("outputs/full_eval/finetuned/metrics.devtest.json",
                             "outputs/full_eval/finetuned_ergative/metrics.json"),
    "translation-only":    ("outputs/ablations/translation_only/metrics.devtest.json",
                             "outputs/ablations/translation_only_ergative/metrics.json"),
    "weight-sweep tw=0.5": ("outputs/ablations/weight_sweep_050/metrics.devtest.json",
                             "outputs/ablations/weight_sweep_050_ergative/metrics.json"),
}

for label, (mpath, epath) in results.items():
    m = json.loads(pathlib.Path(mpath).read_text()) if pathlib.Path(mpath).exists() else {}
    e = json.loads(pathlib.Path(epath).read_text()) if pathlib.Path(epath).exists() else {}
    bleu     = f"{m['bleu']:.2f}"              if m else "n/a"
    chrf     = f"{m['chrf']:.2f}"              if m else "n/a"
    erg_hit  = f"{e['ergative_hit_rate']:.1%}" if e else "n/a"
    erg_err  = f"{e['ergative_error_rate']:.1%}" if e else "n/a"
    print(f"  {label:<25}  BLEU={bleu}  chrF={chrf}  erg_hit={erg_hit}  erg_err={erg_err}")
EOF

echo ""
echo "[ablations] done — outputs in outputs/ablations/"
