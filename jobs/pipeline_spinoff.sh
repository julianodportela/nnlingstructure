#!/bin/bash
# CPU watcher: submits GPU jobs one at a time and waits between them.
# Submit this once and walk away:
#
#   sbatch jobs/pipeline_spinoff.sh
#
# Sequence:
#   [1] annotate_tatoeba  (GPU, ~30 min)   skipped if JSONL already exists
#   [2] train_spinoff     (GPU, up to 23 h)
#   [3] eval_spinoff      (GPU, ~1 h)      submitted at the end; watcher exits
#
#SBATCH --job-name=spinoff_watch
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:59:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/spinoff_watch_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/spinoff_watch_%j.err

SCRATCH=/home/ling3850_jdp69/scratch_ling3850/ling3850_jdp69
ANNOTATED=$SCRATCH/nnling_data/tatoeba_annotated.jsonl

cd /home/ling3850_jdp69/nnlingstructure
mkdir -p jobs/logs

wait_for_job() {
    local jid=$1
    echo "[watch] waiting for job $jid..."
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
    echo "[watch] job $jid finished"
}

# ── Step 1: annotation ───────────────────────────────────────────────────
if [ -f "$ANNOTATED" ]; then
    n=$(wc -l < "$ANNOTATED")
    echo "[1/3] JSONL already exists ($n lines) — skipping annotation"
else
    ANNOT_JID=$(sbatch jobs/annotate_tatoeba.sh | awk '{print $4}')
    echo "[1/3] annotation submitted → job $ANNOT_JID"
    wait_for_job "$ANNOT_JID"

    if [ ! -f "$ANNOTATED" ]; then
        echo "[ERROR] annotation job finished but JSONL not found: $ANNOTATED"
        exit 1
    fi
    echo "[1/3] annotation done — $(wc -l < "$ANNOTATED") lines written"
fi

# ── Step 2: training ─────────────────────────────────────────────────────
TRAIN_JID=$(sbatch jobs/train_spinoff.sh | awk '{print $4}')
echo "[2/3] training submitted → job $TRAIN_JID"
wait_for_job "$TRAIN_JID"

# ── Step 3: eval ─────────────────────────────────────────────────────────
EVAL_JID=$(sbatch jobs/eval_spinoff.sh | awk '{print $4}')
echo "[3/3] eval submitted → job $EVAL_JID"
echo "[watch] pipeline complete — results in outputs/spinoff_eval/ when job $EVAL_JID finishes"
