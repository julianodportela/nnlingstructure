#!/bin/bash
# Pipeline stage 2 (CPU): wait for training run 2, then submit the final eval.
# Expects WATCH_JOB to be set (done automatically by pipeline_1.sh via --export).
#SBATCH --job-name=pipeline_2
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:59:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_2_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/pipeline_2_%j.err

wait_for_job() {
    local jid=$1
    while squeue -j "$jid" --noheader 2>/dev/null | grep -q .; do
        sleep 60
    done
}

cd /home/ling3850_jdp69/nnlingstructure

if [ -z "$WATCH_JOB" ]; then
    echo "[pipeline-2] ERROR: WATCH_JOB not set — was this submitted by pipeline_1.sh?"
    exit 1
fi

# ── Step 5: wait for training run 2 (supertag+deprel, patience=999) ────────
echo "[pipeline-2] waiting for training run 2 (job $WATCH_JOB)..."
wait_for_job "$WATCH_JOB"
echo "[pipeline-2] training run 2 done"

# ── Step 6: submit final eval ───────────────────────────────────────────────
EVAL2_JID=$(sbatch jobs/eval_full.sh | awk '{print $4}')
echo "[pipeline-2] final eval submitted as job $EVAL2_JID"
echo "[pipeline-2] all jobs queued — pipeline complete"
echo "[pipeline-2] results will land in outputs/full_eval/ when job $EVAL2_JID finishes"
