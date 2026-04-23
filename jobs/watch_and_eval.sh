#!/bin/bash
#SBATCH --job-name=watch_and_eval
#SBATCH --partition=education
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=23:00:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/watch_and_eval_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/watch_and_eval_%j.err

WATCH_JOB=9190201

echo "[info] watching for job $WATCH_JOB to finish..."

while squeue -j $WATCH_JOB --noheader 2>/dev/null | grep -q .; do
    sleep 60
done

echo "[info] job $WATCH_JOB is done — submitting eval_full.sh"
cd /home/ling3850_jdp69/nnlingstructure
sbatch jobs/eval_full.sh
