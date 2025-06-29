#!/usr/bin/env python
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


from dataset import (MRIDataset, WSIDataset, MultiModalDataset, 
                    custom_collate_fn_mri, custom_collate_fn_wsi, custom_collate_fn_multimodal)
from model import MRIEDLModel, WSIEDLModel, MultiModalEvidentialModel
from losses import CombinedLoss, MultiModalCombinedLoss
from training import run_training, evaluate_mri, evaluate_wsi, evaluate_multimodal


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_device(config):
    """Setup computation device"""
    if config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def create_datasets_and_loaders(config, clinical_df, train_ids, test_ids):
    """Create datasets and data loaders based on modality"""
    modality = config['modality']
    batch_size = config['training']['batch_size']
    
    print(f"Creating {modality.upper()} datasets...")
    print(f"Train: {len(train_ids)} patients | Test: {len(test_ids)} patients")
    
    if modality == 'mri':
        train_dataset = MRIDataset(
            mri_data_dir=config['data']['mri_data_dir'],
            clinical_df=clinical_df,
            patient_ids=train_ids,
            clinical_features=config['clinical_features'],
            max_slices=config['data']['max_mri_slices']
        )
        
        test_dataset = MRIDataset(
            mri_data_dir=config['data']['mri_data_dir'],
            clinical_df=clinical_df,
            patient_ids=test_ids,
            clinical_features=config['clinical_features'],
            clinical_scaler=train_dataset.clinical_scaler,
            mri_scaler=train_dataset.mri_scaler,
            max_slices=config['data']['max_mri_slices']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_mri)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_mri)
        
    elif modality == 'wsi':
        train_dataset = WSIDataset(
            wsi_features_dir=config['data']['wsi_features_dir'],
            clinical_df=clinical_df,
            patient_ids=train_ids,
            clinical_features=config['clinical_features'],
            max_patches=config['data']['max_wsi_patches']
        )
        
        test_dataset = WSIDataset(
            wsi_features_dir=config['data']['wsi_features_dir'],
            clinical_df=clinical_df,
            patient_ids=test_ids,
            clinical_features=config['clinical_features'],
            clinical_scaler=train_dataset.clinical_scaler,
            max_patches=config['data']['max_wsi_patches']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_wsi)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_wsi)
        
    elif modality == 'multi':
        train_dataset = MultiModalDataset(
            mri_data_dir=config['data']['mri_data_dir'],
            wsi_features_dir=config['data']['wsi_features_dir'],
            clinical_df=clinical_df,
            patient_ids=train_ids,
            clinical_features=config['clinical_features'],
            max_mri_slices=config['data']['max_mri_slices'],
            max_wsi_patches=config['data']['max_wsi_patches']
        )
        
        test_dataset = MultiModalDataset(
            mri_data_dir=config['data']['mri_data_dir'],
            wsi_features_dir=config['data']['wsi_features_dir'],
            clinical_df=clinical_df,
            patient_ids=test_ids,
            clinical_features=config['clinical_features'],
            clinical_scaler=train_dataset.clinical_scaler,
            mri_scaler=train_dataset.mri_scaler,
            max_mri_slices=config['data']['max_mri_slices'],
            max_wsi_patches=config['data']['max_wsi_patches']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_multimodal)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_multimodal)
    
    return train_dataset, test_dataset, train_loader, test_loader


def create_model(config, train_dataset, device):
    """Create model based on modality"""
    modality = config['modality']
    
    if modality == 'mri':
        model = MRIEDLModel(
            mri_dim=train_dataset.n_mri_features,
            clinical_dim=train_dataset.clinical_data.shape[1],
            config=config
        ).to(device)
        
    elif modality == 'wsi':
        model = WSIEDLModel(
            wsi_dim=1024,
            clinical_dim=train_dataset.clinical_data.shape[1],
            config=config
        ).to(device)
        
    elif modality == 'multi':
        model = MultiModalEvidentialModel(
            mri_dim=train_dataset.n_mri_features,
            wsi_dim=1024,
            clinical_dim=train_dataset.clinical_data.shape[1],
            config=config
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    
    return model


def create_loss_function(config):
    """Create loss function based on modality"""
    modality = config['modality']
    
    if modality in ['mri', 'wsi']:
        criterion = CombinedLoss(
            cox_weight=config['loss']['cox_weight'],
            edl_weight=config['loss']['edl_weight'],
            lamb=config['loss']['lamb']
        )
    elif modality == 'multi':
        criterion = MultiModalCombinedLoss(
            cox_weight=config['loss']['cox_weight'],
            edl_weight=config['loss']['edl_weight'],
            focal_weight=config['loss']['focal_weight'],
            lamb=config['loss']['lamb']
        )
    
    return criterion


def main():
    """Main function"""
    print("="*80)
    print("TASK1: Predicting biochemical recurrence (BCR) ")
    print("="*80)
    
    # Load configuration
    config = load_config()
    modality = config['modality']
    print(f" Modality: {modality.upper()}")
    
    # Setup device
    device = setup_device(config)
    
    # Load clinical data
    clinical_df = pd.read_csv(config['data']['clinical_file'])
    patient_ids = clinical_df['patient_id'].values
    labels = clinical_df['BCR'].values
    
    class_counts = np.bincount(labels.astype(int))
    print(f"📋 Dataset: {len(clinical_df)} patients | BCR: {class_counts}")
    
    # Train-test split
    train_ids, test_ids = train_test_split(
        patient_ids, 
        test_size=config['training']['test_size'], 
        stratify=labels, 
        random_state=config['training']['random_state']
    )
    
    # Create datasets and loaders
    train_dataset, test_dataset, train_loader, test_loader = create_datasets_and_loaders(
        config, clinical_df, train_ids, test_ids
    )
    
    # Create model
    model = create_model(config, train_dataset, device)
    
    # Create loss function
    criterion = create_loss_function(config)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config['scheduler']['mode'], 
        factor=config['scheduler']['factor'], 
        patience=config['scheduler']['patience'], 
        verbose=False, 
        min_lr=config['scheduler']['min_lr']
    )
    
    print(f" Training setup: {config['training']['epochs']} epochs | LR: {config['training']['learning_rate']}")
    print("="*80)
    
    # Run training
    history, best_metric, best_epoch = run_training(
        model, train_loader, test_loader, optimizer, criterion, 
        scheduler, config, device, modality
    )
    
    # Load best model and final evaluation
    if config['output']['save_model'] and os.path.exists(config['output']['model_save_path']):
        model.load_state_dict(torch.load(config['output']['model_save_path']))
    
    # Final evaluation
    if modality == 'mri':
        final_results = evaluate_mri(model, test_loader, device)
    elif modality == 'wsi':
        final_results = evaluate_wsi(model, test_loader, device)
    elif modality == 'multi':
        final_results = evaluate_multimodal(model, test_loader, device)
    
    # Print final results
    print(f"\n" + "="*80)
    print(f"TRAINING COMPLETED - {modality.upper()} MODALITY")
    print(f"="*80)
    print(f"Best C-index: {best_metric:.4f} (Epoch {best_epoch})")
    print(f"Total epochs: {len(history)}")
    
    if modality == 'multi':
        print(f"\nFINAL EVALUATION RESULTS:")
        print(f"  MRI Only    - C-index: {final_results['mri_cindex']:.4f} | AUC: {final_results['mri_auc']:.4f} | Accuracy: {final_results['mri_accuracy']:.4f}")
        print(f"  WSI Only    - C-index: {final_results['wsi_cindex']:.4f} | AUC: {final_results['wsi_auc']:.4f} | Accuracy: {final_results['wsi_accuracy']:.4f}")
        print(f"  Multi-modal - C-index: {final_results['fused_cindex']:.4f} | AUC: {final_results['fused_auc']:.4f} | Accuracy: {final_results['fused_accuracy']:.4f}")
        print(f"\nUNCERTAINTY QUANTIFICATION:")
        print(f"  MRI: {final_results['mri_uncertainty']:.4f} | WSI: {final_results['wsi_uncertainty']:.4f} | Fused: {final_results['fused_uncertainty']:.4f}")
    else:
        print(f"\nFINAL EVALUATION RESULTS:")
        print(f"  C-index: {final_results['cindex']:.4f}")
        print(f"  AUC: {final_results['auc']:.4f}")
        print(f"  Accuracy: {final_results['accuracy']:.4f}")
        print(f"  Uncertainty: {final_results['uncertainty']:.4f}")
    
    
    # Save results
    if config['output']['results_save_path'] and history:
        results_df = pd.DataFrame(history)
        results_df.to_csv(config['output']['results_save_path'], index=False)
        print(f"\nResults saved to {config['output']['results_save_path']}")
    
    print("="*80)
    print("Training completed successfully!")
    
    return history, final_results, model


if __name__ == "__main__":
    history, final_results, model = main()