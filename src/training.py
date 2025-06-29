import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

def train_epoch_mri(model, loader, optimizer, criterion, device):
    """Training epoch for MRI model"""
    model.train()
    losses = []
    cox_losses = []
    edl_losses = []
    
    # Progress bar for batches
    batch_pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (slices_list, clinical_batch, labels_batch, times_batch) in enumerate(batch_pbar):
        try:
            batch_size = len(slices_list)
            all_outputs = []
            
            for i in range(batch_size):
                slices = slices_list[i].to(device)
                clinical = clinical_batch[i].to(device)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                    
                outputs = model(slices, clinical)
                all_outputs.append(outputs)
            
            # Combine outputs from batch
            fused_risks = torch.stack([out['fused_risk'] for out in all_outputs])
            pred_dirichlets = torch.stack([out['pred_dirichlet'].squeeze(0) if out['pred_dirichlet'].dim() > 2 else out['pred_dirichlet'] for out in all_outputs])
            
            labels_batch = labels_batch.to(device)
            times_batch = times_batch.to(device)
            
            optimizer.zero_grad()
            
            loss, cox_loss, edl_loss = criterion(
                fused_risks, pred_dirichlets, times_batch, labels_batch, labels_batch
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 10.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                losses.append(loss.item())
                cox_losses.append(cox_loss.item())
                edl_losses.append(edl_loss.item())
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cox': f'{cox_loss.item():.4f}',
                    'edl': f'{edl_loss.item():.4f}'
                })
            
        except Exception as e:
            continue
    
    return (np.mean(losses) if losses else 0, 
            np.mean(cox_losses) if cox_losses else 0, 
            np.mean(edl_losses) if edl_losses else 0)


def train_epoch_wsi(model, loader, optimizer, criterion, device):
    """Training epoch for WSI model - FIXED VERSION"""
    model.train()
    losses = []
    cox_losses = []
    edl_losses = []
    
    # Progress bar for batches
    batch_pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (patches_batch, clinical_batch, labels_batch, times_batch) in enumerate(batch_pbar):
        try:
            batch_size = len(patches_batch)
            all_outputs = []
            
            for i in range(batch_size):
                patches = patches_batch[i].to(device)
                clinical = clinical_batch[i].to(device)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                # DEBUG: Check data shapes
                if batch_idx == 0 and i == 0:
                    print(f"\nDEBUG WSI: patches shape: {patches.shape}, clinical shape: {clinical.shape}")
                    
                outputs = model(patches, clinical)
                all_outputs.append(outputs)
            
            # Combine outputs from batch
            fused_risks = torch.stack([out['fused_risk'] for out in all_outputs])
            pred_dirichlets = torch.stack([out['pred_dirichlet'].squeeze(0) if out['pred_dirichlet'].dim() > 2 else out['pred_dirichlet'] for out in all_outputs])
            
            labels_batch = labels_batch.to(device)
            times_batch = times_batch.to(device)
            
            # DEBUG: Check if we have valid targets
            if batch_idx == 0:
                print(f"DEBUG WSI: labels: {labels_batch}, times: {times_batch}")
                print(f"DEBUG WSI: fused_risks: {fused_risks}")
            
            optimizer.zero_grad()
            
            loss, cox_loss, edl_loss = criterion(
                fused_risks, pred_dirichlets, times_batch, labels_batch, labels_batch
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 10.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                losses.append(loss.item())
                cox_losses.append(cox_loss.item())
                edl_losses.append(edl_loss.item())
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'cox': f'{cox_loss.item():.4f}',
                    'edl': f'{edl_loss.item():.4f}'
                })
            
        except Exception as e:
            print(f"ERROR in WSI training: {e}")
            continue
    
    return (np.mean(losses) if losses else 0, 
            np.mean(cox_losses) if cox_losses else 0, 
            np.mean(edl_losses) if edl_losses else 0)


def train_epoch_multimodal(model, loader, optimizer, criterion, device):
    """Training epoch for multi-modal model"""
    model.train()
    losses = []
    cox_losses = []
    edl_losses = []
    focal_losses = []
    
    # Progress bar for batches
    batch_pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, (mri_batch, wsi_batch, clinical_batch, labels_batch, times_batch) in enumerate(batch_pbar):
        try:
            batch_size = len(mri_batch)
            all_outputs = []
            
            for i in range(batch_size):
                mri_slices = mri_batch[i].to(device)
                wsi_patches = wsi_batch[i].to(device)
                clinical = clinical_batch[i].to(device)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                    
                outputs = model(mri_slices, wsi_patches, clinical)
                all_outputs.append(outputs)
            
            # Combine batch outputs
            mri_risks = torch.stack([out['mri_risk'] for out in all_outputs])
            wsi_risks = torch.stack([out['wsi_risk'] for out in all_outputs])
            fused_risks = torch.stack([out['fused_risk'] for out in all_outputs])
            
            mri_dirichlets = torch.stack([out['mri_dirichlet'].squeeze(0) if out['mri_dirichlet'].dim() > 2 else out['mri_dirichlet'] for out in all_outputs])
            wsi_dirichlets = torch.stack([out['wsi_dirichlet'].squeeze(0) if out['wsi_dirichlet'].dim() > 2 else out['wsi_dirichlet'] for out in all_outputs])
            fused_dirichlets = torch.stack([out['fused_dirichlet'].squeeze(0) if out['fused_dirichlet'].dim() > 2 else out['fused_dirichlet'] for out in all_outputs])
            
            labels_batch = labels_batch.to(device)
            times_batch = times_batch.to(device)
            
            optimizer.zero_grad()
            
            total_loss, cox_loss, edl_loss, focal_loss = criterion(
                mri_risks, wsi_risks, fused_risks,
                mri_dirichlets, wsi_dirichlets, fused_dirichlets,
                times_batch, labels_batch, labels_batch
            )
            
            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss.item() < 20.0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                losses.append(total_loss.item())
                cox_losses.append(cox_loss.item() if isinstance(cox_loss, torch.Tensor) else 0.0)
                edl_losses.append(edl_loss.item() if isinstance(edl_loss, torch.Tensor) else 0.0)
                focal_losses.append(focal_loss.item() if isinstance(focal_loss, torch.Tensor) else 0.0)
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'cox': f'{cox_loss.item():.4f}',
                    'edl': f'{edl_loss.item():.4f}',
                    'focal': f'{focal_loss.item():.4f}'
                })
            
        except Exception as e:
            continue
    
    return (np.mean(losses) if losses else 0, 
            np.mean(cox_losses) if cox_losses else 0, 
            np.mean(edl_losses) if edl_losses else 0,
            np.mean(focal_losses) if focal_losses else 0)


def evaluate_mri(model, loader, device):
    """Evaluation for MRI model"""
    model.eval()
    risks, probs, labels, times = [], [], [], []
    uncertainties = []
    
    with torch.no_grad():
        for slices_list, clinical_batch, labels_batch, times_batch in loader:
            for i in range(len(slices_list)):
                try:
                    slices = slices_list[i].to(device)
                    clinical = clinical_batch[i].to(device)
                    label = labels_batch[i]
                    time = times_batch[i]
                    
                    if clinical.dim() == 1:
                        clinical = clinical.unsqueeze(0)
                    
                    outputs = model(slices, clinical)
                    
                    fused_risk = outputs['fused_risk'].cpu().item()
                    
                    alpha = outputs['pred_dirichlet']
                    S = alpha.sum(dim=1)
                    prob = (alpha[:, 1] / S).cpu().item()
                    uncertainty = (2.0 / S).cpu().item()
                    
                    risks.append(fused_risk)
                    probs.append(prob)
                    uncertainties.append(uncertainty)
                    labels.append(int(label.item()))
                    times.append(time.item())
                    
                except Exception as e:
                    continue
    
    # Calculate metrics
    if len(set(labels)) > 1 and len(risks) > 0:
        try:
            cindex, _, _, _, _ = concordance_index_censored(
                np.array(labels, dtype=bool), np.array(times), np.array(risks))
            auc = roc_auc_score(labels, probs)
            pred_labels = (np.array(probs) > 0.5).astype(int)
            accuracy = accuracy_score(labels, pred_labels)
            mean_uncertainty = np.mean(uncertainties)
        except:
            cindex = auc = accuracy = mean_uncertainty = 0.5
    else:
        cindex = auc = accuracy = mean_uncertainty = 0.5
    
    return {
        'cindex': cindex, 
        'auc': auc, 
        'accuracy': accuracy,
        'uncertainty': mean_uncertainty,
        'n_samples': len(labels)
    }


def evaluate_wsi(model, loader, device):
    """Evaluation for WSI model - FIXED VERSION"""
    model.eval()
    risks, probs, labels, times = [], [], [], []
    uncertainties = []
    
    with torch.no_grad():
        eval_pbar = tqdm(loader, desc="Evaluating", leave=False)
        for patches, clinical, label, time in eval_pbar:
            try:
                patches = patches.squeeze(0).to(device)
                clinical = clinical.squeeze(0).to(device)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                outputs = model(patches, clinical)
                
                fused_risk = outputs['fused_risk'].cpu().item()
                
                alpha = outputs['pred_dirichlet']
                S = alpha.sum(dim=1)
                prob = (alpha[:, 1] / S).cpu().item()
                uncertainty = (2.0 / S).cpu().item()
                
                risks.append(fused_risk)
                probs.append(prob)
                uncertainties.append(uncertainty)
                labels.append(int(label.item()))
                times.append(time.item())
                
            except Exception as e:
                print(f"ERROR in WSI evaluation: {e}")
                continue
    
    # DEBUG: Print evaluation data
    print(f"\nDEBUG EVAL: {len(labels)} samples, unique labels: {set(labels)}")
    print(f"DEBUG EVAL: risks range: {min(risks) if risks else 'N/A'} - {max(risks) if risks else 'N/A'}")
    print(f"DEBUG EVAL: probs range: {min(probs) if probs else 'N/A'} - {max(probs) if probs else 'N/A'}")
    
    # Calculate metrics
    if len(set(labels)) > 1 and len(risks) > 0:
        try:
            cindex, _, _, _, _ = concordance_index_censored(
                np.array(labels, dtype=bool), np.array(times), np.array(risks))
            auc = roc_auc_score(labels, probs)
            pred_labels = (np.array(probs) > 0.5).astype(int)
            accuracy = accuracy_score(labels, pred_labels)
            mean_uncertainty = np.mean(uncertainties)
        except Exception as e:
            print(f"ERROR calculating metrics: {e}")
            cindex = auc = accuracy = mean_uncertainty = 0.5
    else:
        cindex = auc = accuracy = mean_uncertainty = 0.5
    
    return {
        'cindex': cindex, 
        'auc': auc, 
        'accuracy': accuracy,
        'uncertainty': mean_uncertainty,
        'n_samples': len(labels)
    }


def evaluate_multimodal(model, loader, device):
    """Evaluation for multi-modal model"""
    model.eval()
    mri_risks, wsi_risks, fused_risks = [], [], []
    mri_probs, wsi_probs, fused_probs = [], [], []
    mri_uncertainties, wsi_uncertainties, fused_uncertainties = [], [], []
    labels, times = [], []
    
    with torch.no_grad():
        eval_pbar = tqdm(loader, desc="Evaluating", leave=False)
        for batch_data in eval_pbar:
            try:
                if len(batch_data) == 5:
                    mri_batch, wsi_batch, clinical_batch, labels_batch, times_batch = batch_data
                    mri_slices = mri_batch[0].to(device)
                    wsi_patches = wsi_batch[0].to(device)
                    clinical = clinical_batch[0].to(device)
                    label = labels_batch[0]
                    time = times_batch[0]
                else:
                    continue
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                outputs = model(mri_slices, wsi_patches, clinical)
                
                # Extract risks
                mri_risks.append(outputs['mri_risk'].cpu().item())
                wsi_risks.append(outputs['wsi_risk'].cpu().item())
                fused_risks.append(outputs['fused_risk'].cpu().item())
                
                # Extract probabilities and uncertainties
                for alpha, probs_list, uncertainties_list in [
                    (outputs['mri_dirichlet'], mri_probs, mri_uncertainties),
                    (outputs['wsi_dirichlet'], wsi_probs, wsi_uncertainties),
                    (outputs['fused_dirichlet'], fused_probs, fused_uncertainties)
                ]:
                    if alpha.dim() > 2:
                        alpha = alpha.squeeze(0)
                    
                    S = alpha.sum(dim=1)
                    prob = (alpha[:, 1] / S).cpu().item()
                    uncertainty = (2.0 / S).cpu().item()
                    probs_list.append(prob)
                    uncertainties_list.append(uncertainty)
                
                labels.append(int(label.item()))
                times.append(time.item())
                
            except Exception as e:
                continue
    
    # Calculate metrics for all three predictions
    results = {}
    for prefix, risks_list, probs_list, uncertainties_list in [
        ('mri', mri_risks, mri_probs, mri_uncertainties),
        ('wsi', wsi_risks, wsi_probs, wsi_uncertainties),
        ('fused', fused_risks, fused_probs, fused_uncertainties)
    ]:
        if len(set(labels)) > 1 and len(risks_list) > 0:
            try:
                # C-index
                cindex, _, _, _, _ = concordance_index_censored(
                    np.array(labels, dtype=bool), np.array(times), np.array(risks_list))
                
                # AUC
                auc = roc_auc_score(labels, probs_list)
                
                # Accuracy
                pred_labels = (np.array(probs_list) > 0.5).astype(int)
                accuracy = accuracy_score(labels, pred_labels)
                
                mean_uncertainty = np.mean(uncertainties_list)
            except Exception as e:
                cindex = auc = accuracy = mean_uncertainty = 0.5
        else:
            cindex = auc = accuracy = mean_uncertainty = 0.5
        
        results[f'{prefix}_cindex'] = cindex
        results[f'{prefix}_auc'] = auc
        results[f'{prefix}_accuracy'] = accuracy
        results[f'{prefix}_uncertainty'] = mean_uncertainty
    
    results['n_samples'] = len(labels)
    return results


def run_training(model, train_loader, test_loader, optimizer, criterion, scheduler, 
                config, device, modality):
    """Main training loop with tqdm progress bars"""
    
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    best_metric = 0
    best_epoch = 0
    history = []
    patience_counter = 0
    
    # Select appropriate training and evaluation functions
    if modality == 'mri':
        train_fn = train_epoch_mri
        eval_fn = evaluate_mri
        metric_name = 'cindex'
    elif modality == 'wsi':
        train_fn = train_epoch_wsi
        eval_fn = evaluate_wsi
        metric_name = 'cindex'
    elif modality == 'multi':
        train_fn = train_epoch_multimodal
        eval_fn = evaluate_multimodal
        metric_name = 'fused_cindex'
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        # Training
        if modality == 'multi':
            train_loss, cox_loss, edl_loss, focal_loss = train_fn(
                model, train_loader, optimizer, criterion, device
            )
        else:
            train_loss, cox_loss, edl_loss = train_fn(
                model, train_loader, optimizer, criterion, device
            )
            focal_loss = 0.0
        
        # Validation
        val_results = eval_fn(model, test_loader, device)
        
        # Update scheduler
        scheduler.step(val_results[metric_name])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track best model
        if val_results[metric_name] > best_metric:
            best_metric = val_results[metric_name]
            best_epoch = epoch + 1
            patience_counter = 0
            if config['output']['save_model']:
                torch.save(model.state_dict(), config['output']['model_save_path'])
        else:
            patience_counter += 1
        
        # Store history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'cox_loss': cox_loss,
            'edl_loss': edl_loss,
            'focal_loss': focal_loss,
            'learning_rate': current_lr,
            **val_results
        }
        history.append(epoch_data)
        
        # Update epoch progress bar
        if modality == 'multi':
            epoch_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'mri_c': f'{val_results["mri_cindex"]:.3f}',
                'wsi_c': f'{val_results["wsi_cindex"]:.3f}',
                'fused_c': f'{val_results["fused_cindex"]:.3f}',
                'best': f'{best_metric:.3f}'
            })
        else:
            epoch_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'c_index': f'{val_results["cindex"]:.3f}',
                'auc': f'{val_results["auc"]:.3f}',
                'best': f'{best_metric:.3f}'
            })
        
        # Early stopping
        if patience_counter >= patience:
            epoch_pbar.set_description("Early Stopping")
            break
        
        if current_lr <= config['scheduler']['min_lr']:
            epoch_pbar.set_description("Min LR Reached")
            break
    
    epoch_pbar.close()
    return history, best_metric, best_epoch