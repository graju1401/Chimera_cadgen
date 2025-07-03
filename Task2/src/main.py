import os
import yaml
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from dataset import BinaryBCGDataset, custom_collate_fn_wsi
from model import WSIEDLModel
from losses import EDLLoss
from training import run_training, evaluate

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_clinical_data_and_create_splits(config):
    """Load clinical data and create train/test splits"""
    clinical_data_dir = config['data']['clinical_data_dir']
    
    if not os.path.exists(clinical_data_dir):
        raise ValueError(f"Clinical data directory not found: {clinical_data_dir}")
    
    patient_dirs = [d for d in os.listdir(clinical_data_dir) 
                   if os.path.isdir(os.path.join(clinical_data_dir, d))]
    
    clinical_data = []
    
    for patient_id in patient_dirs:
        json_path = os.path.join(clinical_data_dir, patient_id, f"{patient_id}_CD.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    patient_data = json.load(f)
                
                patient_data['patient_id'] = patient_id
                
                if 'BRS' in patient_data and patient_data['BRS'] in ['BRS1', 'BRS2', 'BRS3']:
                    clinical_data.append(patient_data)
                    
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue
    
    if len(clinical_data) == 0:
        raise ValueError("No valid clinical data found")
    
    clinical_df = pd.DataFrame(clinical_data)
    patient_ids = clinical_df['patient_id'].values
    labels = (clinical_df['BRS'] == 'BRS3').astype(int).values
    
    # Calculate class distribution
    class_counts = np.bincount(labels)
    cls_num_list = [class_counts[0], class_counts[1]]
    
    print(f"Dataset: {len(clinical_data)} patients")
    print(f"BRS1/2: {cls_num_list[0]}, BRS3: {cls_num_list[1]}")
    print(f"Class ratio (BRS3/total): {cls_num_list[1]/len(labels):.3f}")
    
    # Stratified split
    try:
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=config['training']['test_size'], 
            stratify=labels, 
            random_state=config['training']['random_state']
        )
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Using random split instead")
        train_ids, test_ids = train_test_split(
            patient_ids, 
            test_size=config['training']['test_size'], 
            random_state=config['training']['random_state']
        )
    
    print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")
    
    return clinical_df, train_ids, test_ids, cls_num_list

def main():
    print("="*60)
    print("BCG Response Classification with MambaFormer + EDL")
    print("="*60)
    
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        return None, None, None
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Load data and create splits
    try:
        clinical_df, train_ids, test_ids, cls_num_list = load_clinical_data_and_create_splits(config)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    # Create datasets and loaders
    print("Creating datasets...")
    
    train_dataset = BinaryBCGDataset(
        wsi_features_dir=config['data']['wsi_features_dir'],
        clinical_data_dir=config['data']['clinical_data_dir'],
        patient_ids=train_ids,
        max_patches=config['data']['max_wsi_patches'],
        training=True,
        augment=True
    )
    
    test_dataset = BinaryBCGDataset(
        wsi_features_dir=config['data']['wsi_features_dir'],
        clinical_data_dir=config['data']['clinical_data_dir'],
        patient_ids=test_ids,
        clinical_scaler=train_dataset.clinical_scaler,
        max_patches=config['data']['max_wsi_patches'],
        training=False,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn_wsi,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=custom_collate_fn_wsi,
        num_workers=0
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Clinical features: {train_dataset.clinical_features.shape[1]} dimensions")
    
    # Create model
    print("Creating model...")
    model = WSIEDLModel(
        wsi_dim=1024,
        clinical_dim=train_dataset.clinical_features.shape[1],
        config=config
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    # Create loss function
    criterion = EDLLoss(
        num_classes=2,
        annealing_step=config['loss']['annealing_step'],
        uncertainty_reg=config['loss']['uncertainty_reg'],
        cls_num_list=cls_num_list,
        beta=config['loss']['beta']
    )
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=config['scheduler']['factor'], 
        patience=config['scheduler']['patience'], 
        min_lr=config['scheduler']['min_lr'],
        verbose=True
    )
    
    # Setup output directories
    if config['output']['save_model']:
        model_dir = os.path.dirname(config['output']['model_save_path'])
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Run training
    try:
        history, best_combined_metric, best_epoch = run_training(
            model, train_loader, test_loader, optimizer, criterion, 
            scheduler, config, device
        )
        
        print(f"\nTraining completed successfully!")
        print(f"Best epoch: {best_epoch}")
        print(f"Best combined metric: {best_combined_metric:.4f}")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        return None, None, None
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return None, None, None
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    try:
        final_results = evaluate(model, test_loader, device)
        
        print(f"\nFinal Results:")
        print(f"F1-score: {final_results['f1_score']:.4f}")
        print(f"AUC: {final_results['auc']:.4f}")
        print(f"Accuracy: {final_results['accuracy']:.4f}")
        print(f"Uncertainty: {final_results['uncertainty']:.4f}")
        print(f"Threshold: {final_results['threshold']:.4f}")
        
        # Print confusion matrix
        if 'predictions' in final_results and 'true_labels' in final_results:
            pred_labels = (np.array(final_results['predictions']) > final_results['threshold']).astype(int)
            cm = confusion_matrix(final_results['true_labels'], pred_labels)
            print(f"\nConfusion Matrix:")
            print(f"{'':>10} {'BRS1/2':>8} {'BRS3':>8}")
            print(f"{'BRS1/2':>10} {cm[0,0]:>8} {cm[0,1]:>8}")
            print(f"{'BRS3':>10} {cm[1,0]:>8} {cm[1,1]:>8}")
            
            # Classification report
            print(f"\nClassification Report:")
            print(classification_report(final_results['true_labels'], pred_labels, 
                                      target_names=['BRS1/2', 'BRS3']))
        
        # Save results
        if config['output']['results_save_path'] and history:
            try:
                results_df = pd.DataFrame(history)
                results_df.to_csv(config['output']['results_save_path'], index=False)
                print(f"\nResults saved to: {config['output']['results_save_path']}")
            except Exception as e:
                print(f"Could not save results: {e}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return history, final_results, model
        
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        return history, None, model

if __name__ == "__main__":
    history, final_results, model = main()