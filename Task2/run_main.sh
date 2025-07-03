#!/bin/bash
#SBATCH --job-name=BCG_Binary
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raju.gudhe@uef.fi

# Load modules and activate environment
module load cuda/12.6
source activate Chimera

# Enable real-time output
export PYTHONUNBUFFERED=1

# Print start time and system info
echo "=================================================================="
echo "BINARY BCG RESPONSE PREDICTION - JOB STARTED"
echo "=================================================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the main training script
echo "Starting BCG Response Prediction Training..."

python src/main.py

# Print completion info
echo "=================================================================="
echo "JOB COMPLETED"
echo "End time: $(date)"
echo "=================================================================="

echo "Training completed. Check results and logs."