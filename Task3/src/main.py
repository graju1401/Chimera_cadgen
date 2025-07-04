import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from local modules
from dataset import MultiModalDataset, multimodal_collate_fn, load_clinical_data_json
from losses import MultiModalCombinedLoss
from model import MultiModalEvidentialModel
from training import train_epoch, evaluate

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_patient_data(data_dir):
    """Get patient IDs and progression events for stratification"""
    patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != 'features']
    patient_ids = []
    progression_events = []
    
    # Load progression status for stratification
    for patient_id in patient_dirs:
        clinical_file = os.path.join(data_dir, patient_id, f"{patient_id}_CD.json")
        if os.path.exists(clinical_file):
            try:
                clinical_data = load_clinical_data_json(clinical_file)
                progression = int(clinical_data.get('progression', 0))
                patient_ids.append(patient_id)
                progression_events.append(progression)
            except:
                continue
    
    return np.array(patient_ids), np.array(progression_events)

def main():
    print("=" * 80)
    print("MultiModal EVIDENTIAL FUSION FOR HR-NMIBC RECURRENCE")
    print("=" * 80)
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    config = load_config(config_path)
    
    # Setup device
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    # Get data configuration
    data_config = config['data']
    training_config = config['training']
    
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Device: {device}")
    
    # Get patient data
    patient_ids, progression_events = get_patient_data(data_config['data_dir'])
    
    # Check class imbalance
    class_counts = np.bincount(progression_events.astype(int))
    #print(f"Dataset: {len(patient_ids)} patients")
    #print(f"Progression distribution: {class_counts} (Event rate: {class_counts[1]/len(progression_events)*100:.1f}%)")
    
    # Train-test split stratified by progression events
    train_ids, test_ids = train_test_split(
        patient_ids, 
        test_size=data_config['test_size'], 
        stratify=progression_events, 
        random_state=data_config['random_state']
    )
    print(f"Train: {len(train_ids)} patients")
    print(f"Test: {len(test_ids)} patients")
    
    # Create datasets
    train_dataset = MultiModalDataset(
        data_config['data_dir'], 
        train_ids, 
        max_wsi_patches=data_config['max_wsi_patches']
    )
    test_dataset = MultiModalDataset(
        data_config['data_dir'], 
        test_ids, 
        train_dataset.clinical_scaler, 
        train_dataset.rna_scaler,
        max_wsi_patches=data_config['max_wsi_patches']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=True, 
        collate_fn=multimodal_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=multimodal_collate_fn
    )
    
    # Initialize model
    model = MultiModalEvidentialModel(
        wsi_dim=config['model']['wsi_encoder']['input_dim'],
        clinical_dim=train_dataset.clinical_features.shape[1],
        config=config
    ).to(device)
    
    # Training setup
    loss_config = config['loss']
    criterion = MultiModalCombinedLoss(
        cox_weight=loss_config['cox_weight'],
        edl_weight=loss_config['edl_weight'],
        focal_weight=loss_config['focal_weight'],
        lamb=loss_config['lamb'],
        focal_alpha=loss_config['focal_alpha'],
        focal_gamma=loss_config['focal_gamma']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    scheduler_config = training_config['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=scheduler_config['mode'], 
        factor=scheduler_config['factor'], 
        patience=scheduler_config['patience'], 
        verbose=True, 
        min_lr=scheduler_config['min_lr']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    #print(f"Clinical features: {train_dataset.clinical_features.shape[1]} (used as K/V in cross-attention)")
    #print(f"RNA-seq: Variable genes per patient")
    #print(f"WSI features: 1024-dim patches")
    #print(f"Architecture: Independent encoders → Clinical K/V cross-attention → EDL")
    #print(f"Primary metric: Concordance Index (C-index)")
    print("=" * 80)
    
    # Training variables
    best_fused_cindex = 0
    best_epoch = 0
    history = []
    patience_counter = 0
    
    # Training loop
    with tqdm(range(training_config['epochs']), desc="Training", ncols=220) as pbar:
        for epoch in pbar:
            # Training
            train_loss, cox_loss, edl_loss, focal_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, training_config['grad_clip']
            )
            
            # Validation
            val_results = evaluate(model, test_loader, device)
            
            # Update scheduler
            scheduler.step(val_results['fused_cindex'])
            current_lr = optimizer.param_groups[0]['lr']
            
            # Track best model
            if val_results['fused_cindex'] > best_fused_cindex:
                best_fused_cindex = val_results['fused_cindex']
                best_epoch = epoch + 1
                patience_counter = 0
                if config['logging']['save_model']:
                    torch.save(model.state_dict(), config['logging']['model_path'])
            else:
                patience_counter += 1
            
            # Store history
            epoch_data = {
                'epoch': epoch + 1,
                'total_loss': train_loss,
                'cox_loss': cox_loss,
                'edl_loss': edl_loss,
                'focal_loss': focal_loss,
                'learning_rate': current_lr,
                **val_results
            }
            history.append(epoch_data)
            
            # Progress bar with C-index focus
            pbar.set_postfix_str(
                f"Loss:{train_loss:.3f}[Cox:{cox_loss:.3f},EDL:{edl_loss:.3f},Focal:{focal_loss:.3f}] | "
                f"RNA C-idx:{val_results['rna_cindex']:.3f} | "
                f"WSI C-idx:{val_results['wsi_cindex']:.3f} | "
                f"Fused C-idx:{val_results['fused_cindex']:.3f} | "
                f"Best:{best_fused_cindex:.3f} | LR:{current_lr:.1e}"
            )
            
            # Detailed metrics every N epochs
            if (epoch + 1) % config['logging']['log_interval'] == 0:
                print(f"\n" + "="*120)
                print(f"EPOCH {epoch + 1} - MultiModal HR-NMIBC EVIDENTIAL LEARNING:")
                print(f"LOSSES: Total={train_loss:.4f} | Cox={cox_loss:.4f} | EDL={edl_loss:.4f} | Focal={focal_loss:.4f}")
                print(f"RNA (Clinical K/V):   C-index={val_results['rna_cindex']:.4f} | AUC={val_results['rna_auc']:.4f} | Uncertainty={val_results['rna_uncertainty']:.4f}")
                print(f"WSI (Clinical K/V):   C-index={val_results['wsi_cindex']:.4f} | AUC={val_results['wsi_auc']:.4f} | Uncertainty={val_results['wsi_uncertainty']:.4f}")
                print(f"MultiModal FUSION:      C-index={val_results['fused_cindex']:.4f} | AUC={val_results['fused_auc']:.4f} | Uncertainty={val_results['fused_uncertainty']:.4f}")
                print("="*120)
            
            # Early stopping
            if patience_counter >= training_config['early_stopping']['patience']:
                pbar.set_description("Early Stop")
                print(f"\n Early stopping - no improvement for {training_config['early_stopping']['patience']} epochs")
                break
            
            if current_lr <= scheduler_config['min_lr']:
                print(f"\n Learning rate reached minimum")
                break
    
    # Load best model and final evaluation
    if os.path.exists(config['logging']['model_path']) and config['logging']['save_model']:
        model.load_state_dict(torch.load(config['logging']['model_path']))
    final_results = evaluate(model, test_loader, device)
    
    # Final results
    print("\n" + "=" * 120)
    print("🏁 MultiModal EVIDENTIAL LEARNING FOR HR-NMIBC COMPLETED")
    print("=" * 120)
    
    print(f" FINAL VALIDATION RESULTS (Best Epoch {best_epoch}):")
    print(f" RNA (Clinical K/V):    C-index={final_results['rna_cindex']:.4f} | AUC={final_results['rna_auc']:.4f}")
    print(f" WSI (Clinical K/V):    C-index={final_results['wsi_cindex']:.4f} | AUC={final_results['wsi_auc']:.4f}")
    print(f" MultiModal FUSION:       C-index={final_results['fused_cindex']:.4f} | AUC={final_results['fused_auc']:.4f}")
    
    print(f"\n EVIDENTIAL UNCERTAINTY QUANTIFICATION:")
    print(f" RNA Uncertainty:       {final_results['rna_uncertainty']:.4f}")
    print(f" WSI Uncertainty:       {final_results['wsi_uncertainty']:.4f}")
    print(f" MultiModal Uncertainty:  {final_results['fused_uncertainty']:.4f}")
    
    # Performance analysis
    rna_improvement = final_results['fused_cindex'] - final_results['rna_cindex']
    wsi_improvement = final_results['fused_cindex'] - final_results['wsi_cindex']
    
    print(f"\n MultiModal EVIDENTIAL FUSION BENEFITS:")
    print(f" C-index improvement over RNA: {rna_improvement:+.4f}")
    print(f" C-index improvement over WSI: {wsi_improvement:+.4f}")
    
    # Save results
    results_df = pd.DataFrame(history)
    results_df.to_csv(config['logging']['results_path'], index=False)
    print(f"\n Results saved to '{config['logging']['results_path']}'")
    
    return history, final_results

if __name__ == "__main__":
    history, results = main()