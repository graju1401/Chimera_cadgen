#!/bin/bash
#SBATCH --job-name=ChimeraSetup
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raju.gudhe@uef.fi

# Load required modules
module load cuda/12.6

# Enable real-time output
export PYTHONUNBUFFERED=1

# Print start time and system info
echo "=================================================================="
echo "CHIMERA ENVIRONMENT SETUP - JOB STARTED"
echo "=================================================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

# Create new conda environment named Chimera
echo "Creating new conda environment 'Chimera'..."
conda create -n Chimera python=3.9 -y

# Activate the environment
echo "Activating Chimera environment..."
source activate Chimera

# Verify environment
echo "=================================================================="
echo "ENVIRONMENT INFO"
echo "=================================================================="
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
echo "=================================================================="

# Upgrade pip first
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first (most important)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
echo "=================================================================="
echo "CUDA VERIFICATION"
echo "=================================================================="
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
echo "=================================================================="

# Install other requirements
echo "Installing remaining packages..."

# State Space Models
echo "Installing mamba-ssm..."
pip install mamba-ssm==2.2.2

# Evidential Deep Learning
echo "Installing edl-pytorch..."
pip install edl-pytorch==0.0.2

# Survival Analysis
echo "Installing survival analysis packages..."
pip install pycox==0.3.0
pip install scikit-survival==0.22.2

# Machine Learning
echo "Installing ML packages..."
pip install scikit-learn==1.3.2
pip install pandas==2.0.3
pip install numpy==1.24.4

# Hyperparameter Optimization
echo "Installing optuna..."
pip install optuna==4.2.1

# Configuration & Utilities
echo "Installing utilities..."
pip install PyYAML==6.0.2
pip install tqdm==4.67.1

# Visualization
echo "Installing visualization packages..."
pip install matplotlib==3.7.5
pip install seaborn==0.13.2

# Medical imaging
echo "Installing medical imaging packages..."
pip install pyradiomics==3.1.0
pip install simpleitk==2.5.0

# System utilities
echo "Installing system utilities..."
pip install psutil==6.0.0

echo "=================================================================="
echo "FINAL VERIFICATION"
echo "=================================================================="
echo "Installed packages:"
pip list | grep -E "(torch|mamba|edl|pycox|scikit|optuna|pyradiomics|simpleitk)"

echo "=================================================================="
echo "TESTING IMPORTS"
echo "=================================================================="
python -c "
import torch
import mamba_ssm
import pandas as pd
import numpy as np
import optuna
import pyradiomics
import SimpleITK
print('✅ All critical packages imported successfully!')
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
"

echo "=================================================================="
echo "CHIMERA ENVIRONMENT SETUP COMPLETED"
echo "End time: $(date)"
echo "=================================================================="
echo ""
echo "🎉 Environment 'Chimera' is ready!"
echo "To use it in future jobs, add: source activate Chimera"
echo "=================================================================="