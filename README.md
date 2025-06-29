# Chimera:  Task 1: Uncertainty-aware Multi-Modal MambaFormer for Biochemical Recurrence Prediction

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art multi-modal deep learning framework combining **MRI radiomics**, **whole slide imaging (WSI)**, and **clinical data** for predicting biochemical recurrence (BCR) in prostate cancer patients. The model leverages **Mamba state-space models** with **transformer attention** and **evidential deep learning** for uncertainty quantification.

## 🚀 Features

- **Multi-Modal Fusion**: Integrates MRI radiomics, pathology WSI features, and clinical data
- **MambaFormer Architecture**: Novel combination of Mamba SSMs and Transformer attention
- **Evidential Deep Learning**: Uncertainty quantification using Dirichlet distributions
- **Survival Analysis**: Cox proportional hazards modeling for time-to-event prediction
- **Hyperparameter Optimization**: Multi-objective Optuna optimization (C-index & AUC)
- **Automated Feature Extraction**: Radiomic feature extraction from raw MRI images
- **SLURM Integration**: Ready-to-use SBATCH scripts for HPC clusters

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Quick Start](#quick-start)
- [Training](#training)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Inference](#inference)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- CUDA 12.1+ (for GPU support)
- Conda/Miniconda

### Option 1: Using Conda Environment (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/chimera-bcr-prediction.git
cd chimera-bcr-prediction

# Create and activate conda environment
conda create -n Chimera python=3.9 -y
conda activate Chimera

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using SLURM Setup Script

For HPC clusters with SLURM:

```bash
# Submit environment setup job
sbatch install_requirements.sh
```

### Requirements

```txt
# Core packages
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
mamba-ssm==2.2.2
edl-pytorch==0.0.2
pycox==0.3.0
scikit-survival==0.22.2
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.4
optuna==4.2.1
PyYAML==6.0.2
tqdm==4.67.1
matplotlib==3.7.5
seaborn==0.13.2
pyradiomics==3.1.0
simpleitk==2.5.0
psutil==6.0.0
```

## 📁 Project Structure

```
chimera-bcr-prediction/
├── src/                              # Source code
│   ├── main.py                       # Main training script
│   ├── inference.py                  # Inference pipeline
│   ├── model.py                      # MambaFormer model architectures
│   ├── dataset.py                    # Data loading and preprocessing
│   ├── losses.py                     # Loss functions (Cox, EDL, Focal)
│   ├── training.py                   # Training and evaluation loops
│   ├── optuna_tuning.py              # Hyperparameter optimization
│   └── mpMRI_slide_level_radiomic_feature_extraction.py  # Feature extraction
├── config.yaml                      # Configuration file
├── requirements.txt                  # Python dependencies
├── install_requirements.sh             # Environment setup script
├── run_main.sh                      # Training SBATCH script
├── run_optuna.sh                    # Optimization SBATCH script
├── run_inference.sh                 # Inference SBATCH script
└── README.md                        # This file
```

## 📊 Data Format

### Expected Directory Structure

```
data/
├── clinical_data.csv                # Clinical features
├── radiology/
│   ├── images/                      # Raw MRI images
│   │   ├── patient_001/
│   │   │   ├── image_001_t2w.mha
│   │   │   ├── image_001_adc.mha
│   │   │   ├── image_001_hbv.mha
│   │   │   └── image_001_mask.mha
│   │   └── patient_002/
│   │       └── ...
│   └── radiomic_features/           # Extracted features (auto-generated)
│       ├── patient_001/
│       └── patient_002/
└── pathology/
    └── features/
        └── features/                # WSI features (.pt files)
            ├── patient_001/
            │   └── slide_001.pt
            └── patient_002/
                └── slide_002.pt
```

### Clinical Data Format (CSV)

```csv
patient_id,age_at_prostatectomy,primary_gleason,secondary_gleason,BCR,time_to_follow-up/BCR,...
patient_001,65,3,4,0,24.5,...
patient_002,72,4,5,1,12.3,...
```

### Inference Data Format (JSON per patient)

```json
{
  "age_at_prostatectomy": 66,
  "primary_gleason": 3,
  "secondary_gleason": 4,
  "tertiary_gleason": 5,
  "ISUP": 2,
  "pre_operative_PSA": 8.3,
  "BCR": "1.0",
  "time_to_follow-up/BCR": 1.3,
  "positive_lymph_nodes": "1",
  "capsular_penetration": "1",
  "positive_surgical_margins": 1,
  "invasion_seminal_vesicles": "1",
  "lymphovascular_invasion": "1.0",
  "earlier_therapy": "none"
}
```

## 🚀 Quick Start

### 1. Configure the Model

Edit `config.yaml` to set your modality and data paths:

```yaml
modality: 'multi'  # Options: 'mri', 'wsi', 'multi'

data:
  mri_data_dir: "data/radiology/radiomic_features"
  wsi_features_dir: "data/pathology/features/features"
  clinical_file: "data/clinical_data.csv"
```

### 2. Train the Model

**Local Training:**
```bash
python src/main.py
```

**SLURM Cluster:**
```bash
sbatch run_main.sh
```

### 3. Run Inference

**Local Inference:**
```bash
python src/inference.py --config config.yaml --model best_model.pth --data data/
```

**SLURM Cluster:**
```bash
sbatch run_inference.sh
```

## 🎯 Training

The training pipeline supports three modalities:

- **MRI Only**: `modality: 'mri'`
- **WSI Only**: `modality: 'wsi'`
- **Multi-Modal**: `modality: 'multi'` (recommended)

### Key Features

- **Automatic Radiomic Extraction**: Extracts 100+ radiomic features from raw MRI
- **Multi-Modal Attention**: Cross-modal attention between imaging and clinical data
- **Evidential Learning**: Uncertainty quantification for predictions
- **Survival Analysis**: Cox regression for time-to-event modeling

### Training Outputs

- `best_model.pth`: Best model weights
- `training_results.csv`: Training history and metrics
- Console logs with real-time metrics

## 🔧 Hyperparameter Optimization

Run multi-objective optimization (C-index & AUC):

```bash
# Local optimization
python src/optuna_tuning.py --modality multi --n_trials 100

# SLURM cluster
sbatch run_optuna.sh
```

### Optimization Features

- **Multi-Objective**: Simultaneously optimizes C-index and AUC
- **Pareto Frontier**: Finds optimal trade-offs between objectives
- **Auto-Config Generation**: Creates optimized configuration files
- **SQLite Storage**: Persistent study storage for resume capability

### Output Files

- `config_best_cindex.yaml`: Configuration with best C-index
- `config_best_auc.yaml`: Configuration with best AUC
- `pareto_configs/`: Multiple Pareto-optimal configurations

## 🔮 Inference

The inference pipeline automatically:

1. **Extracts radiomic features** from raw MRI images
2. **Loads clinical data** from JSON files
3. **Prepares pathology features** from .pt files
4. **Runs multi-modal prediction**
5. **Generates comprehensive metrics**

### Inference Outputs

```
📊 RESULTS
================================================================================
FUSED - C-index: 0.7416 | AUC: 0.7723 | Acc: 0.7158
MRI   - C-index: 0.3514 | AUC: 0.7642 | Acc: 0.7158
WSI   - C-index: 0.3891 | AUC: 0.7522 | Acc: 0.7684
```

- `inference_results.csv`: Detailed predictions per patient
- Individual modality predictions with uncertainty estimates

## ⚙️ Configuration

### Key Configuration Options

```yaml
# Model Architecture
model:
  d_model: 256              # Model dimension
  n_layers: 5               # Number of MambaFormer layers
  dropout: 0.1              # Dropout rate
  max_slices_attention: 1000   # Max MRI slices for attention
  max_patches_attention: 5000  # Max WSI patches for attention

# Training Parameters
training:
  epochs: 50
  batch_size: 2
  learning_rate: 0.0001
  weight_decay: 0.00001

# Loss Weights
loss:
  cox_weight: 1.0           # Survival loss weight
  edl_weight: 0.5           # Evidential loss weight
  focal_weight: 0.4         # Focal loss weight (multi-modal only)
```

## 📈 Results

### Performance Metrics

The model evaluates performance using multiple metrics:

- **C-index**: Concordance index for survival analysis
- **AUC**: Area under the ROC curve
- **Accuracy**: Classification accuracy
- **Precision/Recall/F1**: Additional classification metrics
- **Uncertainty**: Mean epistemic uncertainty from EDL

### Expected Performance

On typical prostate cancer BCR datasets:
- **C-index**: 0.75-0.85
- **AUC**: 0.80-0.90
- **Multi-modal improvement**: 5-10% over single modalities

## 🔬 Model Architecture

### MambaFormer Components

1. **MRI Encoder**: Processes radiomic features using Mamba SSM
2. **WSI Encoder**: Processes pathology patches using Mamba SSM
3. **Clinical Encoder**: Encodes clinical variables
4. **Cross-Attention**: Attends between modalities and clinical data
5. **EDL Heads**: Evidential layers for uncertainty quantification
6. **Fusion**: Dempster-Shafer combination of predictions

### Key Innovations

- **State-Space Models**: Efficient sequence modeling with Mamba
- **Evidential Learning**: Principled uncertainty quantification
- **Multi-Modal Fusion**: Cross-attention between imaging modalities
- **Survival Integration**: Cox regression within deep learning framework

## 🖥️ SLURM Usage

### Environment Setup
```bash
sbatch setup_chimera_env.sh
```

### Training Jobs
```bash
# Standard training
sbatch run_main.sh

# Hyperparameter optimization
sbatch run_optuna.sh

# Inference
sbatch run_inference.sh
```

### Resource Requirements

- **Training**: 16-32GB RAM, 1 GPU, 8 CPUs
- **Optimization**: 64GB RAM, 1 GPU, 8 CPUs, 48 hours
- **Inference**: 16GB RAM, 1 GPU (optional), 8 CPUs



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{chimera2024,
  title={Uncertainty-aware Multi-Modal MambaFormer for Biochemical Recurrence Prediction in Prostate Cancer},
  author={Raju Gudhe, Hesam Hakimnejad, Pekka Ruusuvuori, Minna Kaikkonen-Määttä},
  journal={Journal Name},
  year={2024},
  volume={XX},
  pages={XXX-XXX}
}
```

## 🙏 Acknowledgments

- [Mamba SSM](https://github.com/state-spaces/mamba) for state-space models
- [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics) for radiomic feature extraction
- [Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning) for uncertainty quantification
- [PyCox](https://github.com/havakv/pycox) for survival analysis

## 📞 Contact

- **Author**: Raju Gudhe
- **Email**: raju.gudhe@uef.fi
- **Institution**: University of Eastern Finland
- **Project Link**: [https://github.com/yourusername/chimera-bcr-prediction](https://github.com/yourusername/chimera-bcr-prediction)

---

**⭐ Star this repository if it helps your research!**
