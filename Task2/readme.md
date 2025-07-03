# BCG Response Prediction with Multi-Modal Deep Learning

## Overview

This repository contains a multi-modal deep learning pipeline for predicting Bacillus Calmette-Guérin (BCG) response subtypes in high-risk non-muscle-invasive bladder cancer (HR-NMIBC) patients. The model combines H&E-stained histopathology slides with clinical data to predict BCG response subtypes (BRS1, BRS2, BRS3) defined using a validated biomarker signature.

### Key Features

- **Multi-modal Architecture**: Combines WSI (Whole Slide Image) features with clinical data
- **MambaFormer Architecture**: Novel hybrid architecture combining Mamba state-space models with transformer attention
- **Evidential Deep Learning (EDL)**: Provides uncertainty quantification for predictions
- **Binary Classification**: Clinically relevant BRS3 vs BRS1/2 classification
- **Attention Mechanisms**: Cross-attention between WSI and clinical features
- **Data Augmentation**: Specialized augmentation for both WSI patches and clinical features

## Architecture

The pipeline consists of several key components:

1. **WSI Patch Encoder**: Processes 1024-dimensional patch features from histopathology slides
2. **Clinical Encoder**: Processes structured clinical data
3. **MambaFormer Layers**: Hybrid sequence modeling with cross-attention
4. **Attention Aggregator**: Aggregates patch-level features to patient-level representation
5. **EDL Classifier**: Provides predictions with uncertainty quantification

## Dataset Structure

```
data/
├── data/                          # Clinical data and images
│   ├── 2A_001/
│   │   ├── 2A_001_CD.json        # Clinical data
│   │   ├── 2A_001_HE.tif         # H&E slide
│   │   └── 2A_001_HE_mask.tif    # Tissue mask
│   ├── 2A_002/
│   │   └── ...
│   └── ... (181 patients total)
└── features/
    ├── coordinates/               # Patch coordinates
    │   ├── 2A_001_HE.npy
    │   └── ... (181 files)
    └── features/                  # Extracted patch features
        ├── 2A_001_HE.pt          # 1024-dim features per patch
        └── ... (181 files)
```

### Clinical Data Format

Each `*_CD.json` file contains:
- Patient demographics (age, sex)
- Tumor characteristics (stage, grade, recurrence status)
- Treatment history (reTUR, instillations)
- Risk stratification (EORTC)
- BCG response subtype (BRS1, BRS2, BRS3)



## Usage

### Configuration

Edit `config.yaml` to adjust training parameters:

```yaml
device: 'cuda'
data:
  clinical_data_dir: 'data/data'
  wsi_features_dir: 'data/features/features'
  max_wsi_patches: 1000

model:
  d_model: 64
  n_layers: 2
  n_heads: 4
  dropout: 0.2

training:
  batch_size: 8
  epochs: 40
  learning_rate: 0.001
  early_stopping_patience: 15
```

### Training

#### Local Training
```bash
python src/main.py
```

#### SLURM Cluster Training
```bash
sbatch run_main.sh
```

### Inference

#### Local Inference
```bash
python src/inference.py
```

#### SLURM Cluster Inference
```bash
sbatch run_inference.sh
```

## Model Components

### 1. WSI Processing (`dataset.py`)

- **Patch Augmentation**: Gaussian noise, feature dropout
- **Clinical Augmentation**: Noise injection for continuous features
- **Data Scaling**: RobustScaler for clinical features
- **Class Balancing**: Enhanced augmentation for minority classes

### 2. Model Architecture (`model.py`)

- **PatchEncoder**: Encodes 1024-dim patch features to embedding space
- **ClinicalEncoder**: Processes structured clinical data
- **MambaTransformerLayer**: Hybrid sequence modeling with cross-attention
- **AttentionAggregator**: Patient-level feature aggregation
- **EDLClassifier**: Evidential classifier with uncertainty quantification

### 3. Training (`training.py`)

- **EDL Loss**: Combines classification loss with uncertainty regularization
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Prevents overfitting
- **Metric Optimization**: F1-score and AUC optimization

### 4. Inference (`inference.py`)

- **Uncertainty Quantification**: Provides prediction confidence
- **Threshold Optimization**: Finds optimal classification threshold
- **Comprehensive Metrics**: AUC, F1-score, precision, recall

## Key Files

```
src/
├── main.py           # Main training script
├── inference.py      # Inference and evaluation
├── model.py          # Model architectures
├── dataset.py        # Data loading and augmentation
├── training.py       # Training utilities
├── losses.py         # Loss functions (EDL, Focal)
└── config.yaml       # Configuration file
```

## Output Files

- `best_model.pth`: Trained model weights
- `training_results.csv`: Training history and metrics
- `inference_results.csv`: Detailed inference results with uncertainties
- `*.out` and `*.err`: SLURM job outputs

## Performance Metrics

The model is evaluated using:
- **AUC**: Area under the ROC curve
- **F1-Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall classification accuracy
- **Uncertainty**: Mean prediction uncertainty
- **Confusion Matrix**: Detailed classification breakdown

## Clinical Relevance

- **BRS3 vs BRS1/2**: Clinically relevant binary classification
- **Treatment Decision Support**: Helps identify patients likely to respond to BCG therapy
- **Uncertainty Quantification**: Provides confidence estimates for clinical decision-making

## Advanced Features

### Evidential Deep Learning (EDL)
- Provides uncertainty estimates alongside predictions
- Handles epistemic and aleatoric uncertainty
- Useful for clinical decision-making under uncertainty

### MambaFormer Architecture
- Combines Mamba state-space models with transformer attention
- Efficient sequence modeling for variable-length patch sequences
- Cross-attention between WSI and clinical modalities

### Data Augmentation
- Specialized augmentation for histopathology patches
- Clinical feature augmentation with domain knowledge
- Enhanced augmentation for minority classes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `max_wsi_patches`
2. **Mamba Import Error**: Install `mamba-ssm` or model falls back to transformer
3. **File Not Found**: Check data paths in `config.yaml`
4. **Training Instability**: Adjust learning rate or increase gradient clipping

### Performance Optimization

- Use GPU for training (set `device: 'cuda'`)
- Adjust `max_wsi_patches` based on available memory
- Use `pin_memory=True` for faster data loading
- Enable mixed precision training for larger models

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For questions or issues, please contact:
- Email: raju.gudhe@uef.fi
- GitHub Issues: [Project Issues](https://github.com/yourusername/bcg-response-prediction/issues)

## Acknowledgments
- Mamba SSM implementation from state-spaces/mamba
- Evidential Deep Learning implementation inspired by sensable/EDL-pytorch
