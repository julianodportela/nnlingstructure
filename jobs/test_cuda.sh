#!/bin/bash
#SBATCH --job-name=cuda_test
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --output=/home/ling3850_jdp69/nnlingstructure/jobs/logs/cuda_test_%j.out
#SBATCH --error=/home/ling3850_jdp69/nnlingstructure/jobs/logs/cuda_test_%j.err

cd /home/ling3850_jdp69/nnlingstructure

module load CUDA/12.8.0
source .venv/bin/activate

python -c "
import torch, sys
print(f'torch: {torch.__version__}')
print(f'cuda compiled: {torch.version.cuda}')
print(f'cuda available: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('[FAIL] GPU not reachable — do not submit the 23-hour job')
    sys.exit(1)
print(f'[PASS] device: {torch.cuda.get_device_name(0)}')
print('[PASS] Safe to submit train.sh')
"
