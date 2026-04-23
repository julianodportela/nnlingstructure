#!/bin/bash
# Pipeline stage 1 (CPU): wait for training run 1, run eval, then launch training run 2
# and hand off to pipeline_2.sh.
#SBATCH --job-name=pipeline_1
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_1_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_1_%j.err

wait_for_job() {
    local jid=$1
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
}

cd /home/ling3850_jdp69/nnlingstructure

# ── Step 1: wait for training run 1 (old supertag format, patience=5) ─────
echo "[pipeline-1] waiting for training run 1 (job 9190201)..."
wait_for_job 9190201
echo "[pipeline-1] training run 1 done"

# ── Step 2: evaluate run 1 ─────────────────────────────────────────────────
EVAL1_JID=$(sbatch jobs/eval_full.sh | awk '{print $4}')
echo "[pipeline-1] eval run 1 submitted as job $EVAL1_JID"
wait_for_job "$EVAL1_JID"
echo "[pipeline-1] eval run 1 done"

# Preserve run 1 outputs before the next eval overwrites them.
[ -d outputs/full_eval ] && mv outputs/full_eval outputs/full_eval_run1
echo "[pipeline-1] outputs saved to outputs/full_eval_run1/"

# ── Step 3: submit training run 2 (supertag+deprel, patience=999) ──────────
TRAIN2_JID=$(sbatch jobs/train.sh | awk '{print $4}')
echo "[pipeline-1] training run 2 submitted as job $TRAIN2_JID"

# ── Step 4: hand off to pipeline_2 ────────────────────────────────────────
sbatch --export=ALL,WATCH_JOB="$TRAIN2_JID" jobs/pipeline_2.sh
echo "[pipeline-1] pipeline_2 submitted — done"
