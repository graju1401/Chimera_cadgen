# Multimodal Evidential Learning for HR-NMIBC Recurrence Prediction

## 🔬 Overview

This repository implements a multimodal evidential learning framework for predicting recurrence in High-Risk Non-Muscle Invasive Bladder Cancer (HR-NMIBC) patients. The model integrates **histopathology**, **RNA-seq transcriptomics**, and **clinical data** to predict patient-level time-to-recurrence using both morphological and molecular information.

### Key Features
- **Multimodal Fusion**: Combines histopathology (WSI), RNA-seq, and clinical data
- **Evidential Learning**: Quantifies prediction uncertainty using Dirichlet distributions
- **Mamba-Transformer Architecture**: Efficient processing of variable-length sequences
- **Survival Analysis**: Cox proportional hazards model for time-to-event prediction
- **Cross-Attention**: Clinical features guide RNA and WSI feature extraction

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │  Clinical Data  │
                    │   (K/V pairs)   │
                    └─────────┬───────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  RNA-seq     │  │ Histopathology│  │   Clinical   │
    │  Encoder     │  │   WSI Encoder │  │   Encoder    │
    │  (Mamba)     │  │   (Mamba)     │  │              │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘
           │                 │
           ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │Cross-Attention│  │Cross-Attention│
    │ (Clinical K/V)│  │ (Clinical K/V)│
    └──────┬───────┘  └──────┬───────┘
           │                 │
           ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │ Evidential   │  │ Evidential   │
    │ Classifier   │  │ Classifier   │
    │ (Dirichlet)  │  │ (Dirichlet)  │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └─────────┬───────┘
                     ▼
           ┌──────────────────┐
           │ Dempster-Shafer  │
           │ Fusion           │
           │ (DS Combination) │
           └──────┬───────────┘
                  ▼
           ┌──────────────────┐
           │ Final Prediction │
           │ (Risk + Uncertainty)│
           └──────────────────┘
```

## 📊 Task Objective

This implementation addresses Multimodal integration for HR-NMIBC recurrence prediction by:

1. **Morphological Analysis**: Processing H&E histopathology slides to extract spatial tumor patterns
2. **Molecular Profiling**: Analyzing RNA-seq transcriptomics from selected tumor regions
3. **Clinical Integration**: Incorporating patient demographics and clinical variables
4. **Uncertainty Quantification**: Providing confidence estimates for clinical decision-making
5. **Time-to-Event Modeling**: Predicting both recurrence probability and timing

> **Note**: RNA-seq data is derived from a selected tumor region within the histopathology slide, ensuring spatial correspondence between molecular and morphological features.

## 🗂️ Data Structure

```
data/
├── data/
│   ├── task3_quality_control.csv          # Quality control metadata
│   └── {patient_id}/
│       ├── {patient_id}_CD.json           # Clinical data (demographics, staging, etc.)
│       ├── {patient_id}_HE.tif            # H&E histopathology slide
│       ├── {patient_id}_HE_mask.tif       # Tumor region mask
│       └── {patient_id}_RNA.json          # RNA-seq expression data
└── features/
    ├── coordinates/
    │   └── {patient_id}_HE.npy             # Patch coordinates
    └── features/
        └── {patient_id}_HE.pt              # Pre-extracted histopathology features
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install mamba-ssm
pip install edl-pytorch
pip install scikit-survival
pip install pycox
pip install pandas numpy scikit-learn
pip install pyyaml tqdm
```

### Training
```bash
# Local training
python src/main.py

# HPC/Slurm training
sbatch run_main.sh
```

### Inference
```bash
# Local inference
python src/inference.py

# HPC/Slurm inference  
sbatch run_inference.sh
```

## 📁 Repository Structure

```
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── dataset.py               # Multimodal data loading & preprocessing
│   ├── model.py                 # MambaFormer evidential architecture
│   ├── losses.py                # Combined loss functions (Cox + EDL + Focal)
│   ├── training.py              # Training and evaluation loops
│   ├── main.py                  # Main training script
│   └── inference.py             # Inference evaluation script
├── config.yaml                  # Configuration parameters
├── run_main.sh                  # Slurm training script
├── run_inference.sh             # Slurm inference script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```


## 🔬 Model Components

### 1. RNA-seq Encoder (RNAMambaFormer)
- **Input**: Variable-length gene expression vectors
- **Architecture**: Mamba + Cross-attention with clinical features
- **Output**: Patient-level RNA embedding (64-dim)

### 2. Histopathology Encoder (WSIMambaFormer)  
- **Input**: Variable-length WSI patch features (1024-dim)
- **Architecture**: Mamba + Cross-attention with clinical features
- **Output**: Patient-level WSI embedding (64-dim)

### 3. Clinical Encoder
- **Input**: 13 clinical features (age, sex, staging, etc.)
- **Processing**: Layer normalization + MLP
- **Role**: Provides key/value pairs for cross-attention

### 4. Evidential Learning Heads
- **Classification**: Dirichlet distribution (uncertainty quantification)
- **Regression**: Normal-Inverse-Gamma distribution (risk scores)
- **Fusion**: Dempster-Shafer combination rule

## 📈 Evaluation Metrics

### Primary Metrics
- **C-Index**: Concordance index for survival analysis
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Uncertainty**: Mean evidential uncertainty from Dirichlet distributions

### Additional Metrics
- Accuracy, Precision, Recall, F1-Score
- Specificity, Confusion Matrix
- Multimodal improvement analysis

## 🎯 Results Interpretation

The model outputs:
1. **Risk Scores**: Continuous values for time-to-recurrence
2. **Probabilities**: Discrete recurrence predictions (0-1)
3. **Uncertainties**: Confidence estimates for each prediction
4. **Modality Contributions**: Individual RNA, WSI, and fused predictions


### Hyperparameter Tuning
- Adjust learning rates and batch sizes in `config.yaml`
- Modify architecture parameters (d_model, n_layers, dropout)
- Tune loss weights (cox_weight, edl_weight, focal_weight)

## 🐛 Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `max_wsi_patches` or `batch_size`
2. **Missing Data**: Check data directory structure
3. **Convergence Issues**: Adjust learning rate or loss weights

### Memory Optimization
- Enable gradient checkpointing for large models
- Use mixed precision training (FP16)
- Process data in smaller batches


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



## 📞 Contact

For questions or issues, please open a GitHub issue or contact [raju.gudhe@uef.fi](mailto:raju.gudhe@uef.fi).


