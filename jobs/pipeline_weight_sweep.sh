#!/bin/bash
# Wait for the ablations pipeline to finish, then run the weight-sweep training
# and eval that couldn't be submitted due to the per-user job limit.
#SBATCH --job-name=pipeline_weight_sweep
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:59:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_weight_sweep_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_weight_sweep_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69

wait_for_job() {
    local jid=$1
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
}

cd /home/ling3850_jdp69/nnlingstructure

# ── Wait for the ablations pipeline (handles trans-only) ───────────────────
echo "[sweep] waiting for pipeline_ablations job ${ABLATIONS_JID}..."
wait_for_job "$ABLATIONS_JID"
echo "[sweep] pipeline_ablations done"

# ── Submit weight-sweep training ───────────────────────────────────────────
SWEEP_JID=$(sbatch jobs/train_weight_sweep.sh | awk '{print $4}')
echo "[sweep] weight-sweep training: job $SWEEP_JID"

if [ -z "$SWEEP_JID" ]; then
    echo "[sweep] ERROR: failed to submit training job"
    exit 1
fi

# ── Wait for training ──────────────────────────────────────────────────────
echo "[sweep] waiting for training job $SWEEP_JID..."
wait_for_job "$SWEEP_JID"
echo "[sweep] training done"

SWEEP_LOG="jobs/logs/train_weight_sweep_${SWEEP_JID}.out"
SWEEP_CKPT=$(grep "^CKPT_PATH=" "$SWEEP_LOG" 2>/dev/null | tail -1 | cut -d= -f2-)
if [ -z "$SWEEP_CKPT" ]; then
    SWEEP_CKPT=$(find $SCRATCH/nllb_checkpoints -maxdepth 2 -name best -type d 2>/dev/null \
        | grep weight50 | sort | tail -1)
fi
echo "[sweep] checkpoint: $SWEEP_CKPT"

if [ -z "$SWEEP_CKPT" ] || [ ! -d "$SWEEP_CKPT" ]; then
    echo "[sweep] ERROR: checkpoint not found — aborting eval"
    exit 1
fi

# ── Submit eval ────────────────────────────────────────────────────────────
EVAL_JID=$(sbatch jobs/eval_ablation.sh "$SWEEP_CKPT" weight_sweep_050 | awk '{print $4}')
echo "[sweep] eval: job $EVAL_JID"

wait_for_job "$EVAL_JID"
echo "[sweep] eval done"

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " ABLATION SUMMARY"
echo "════════════════════════════════════════════════"

python3 - <<'EOF'
import json, pathlib

results = {
    "baseline":            ("outputs/full_eval/baseline/metrics.devtest.json",
                             "outputs/full_eval/baseline_ergative/metrics.json"),
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
    bleu    = f"{m['bleu']:.2f}"               if m else "n/a"
    chrf    = f"{m['chrf']:.2f}"               if m else "n/a"
    erg_hit = f"{e['ergative_hit_rate']:.1%}"  if e else "n/a"
    erg_err = f"{e['ergative_error_rate']:.1%}" if e else "n/a"
    print(f"  {label:<25}  BLEU={bleu}  chrF={chrf}  erg_hit={erg_hit}  erg_err={erg_err}")
EOF

echo ""
echo "[sweep] done — outputs in outputs/ablations/"
