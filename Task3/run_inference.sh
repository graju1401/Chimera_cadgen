#!/bin/bash
#SBATCH --job-name=ChimeraT1 
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raju.gudhe@uef.fi

# Load modules and activate environment
module load cuda/12.6
source activate Chimera2

# Enable real-time output
export PYTHONUNBUFFERED=1

# Print start time and system info
echo "=================================================================="
echo "MULTI-PARAMETRIC MAMBA ATTENTION MIL - JOB STARTED"
echo "=================================================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=================================================================="

# Run the script with real-time output
python src/inference.py

# Print completion info
echo "=================================================================="
echo "JOB COMPLETED"
echo "End time: $(date)"
echo "=================================================================="
