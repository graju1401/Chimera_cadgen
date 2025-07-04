import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sksurv.metrics import concordance_index_censored

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=0.5):
    model.train()
    losses = []
    cox_losses = []
    edl_losses = []
    focal_losses = []
    
    for batch_idx, (rna_batch, wsi_batch, clinical_batch, labels_batch, times_batch) in enumerate(loader):
        try:
            batch_size = len(rna_batch)
            all_outputs = []
            
            for i in range(batch_size):
                gene_expressions = rna_batch[i].to(device)
                wsi_patches = wsi_batch[i].to(device)
                clinical = clinical_batch[i].to(device)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                    
                outputs = model(gene_expressions, wsi_patches, clinical)
                all_outputs.append(outputs)
            
            # Combine batch outputs
            rna_risks = torch.stack([out['rna_risk'] for out in all_outputs])
            wsi_risks = torch.stack([out['wsi_risk'] for out in all_outputs])
            fused_risks = torch.stack([out['fused_risk'] for out in all_outputs])
            
            rna_dirichlets = torch.stack([out['rna_dirichlet'].squeeze(0) if out['rna_dirichlet'].dim() > 2 else out['rna_dirichlet'] for out in all_outputs])
            wsi_dirichlets = torch.stack([out['wsi_dirichlet'].squeeze(0) if out['wsi_dirichlet'].dim() > 2 else out['wsi_dirichlet'] for out in all_outputs])
            fused_dirichlets = torch.stack([out['fused_dirichlet'].squeeze(0) if out['fused_dirichlet'].dim() > 2 else out['fused_dirichlet'] for out in all_outputs])
            
            labels_batch = labels_batch.to(device)
            times_batch = times_batch.to(device)
            
            optimizer.zero_grad()
            
            # Calculate loss
            total_loss, cox_loss, edl_loss, focal_loss = criterion(
                rna_risks, wsi_risks, fused_risks,
                rna_dirichlets, wsi_dirichlets, fused_dirichlets,
                times_batch, labels_batch, labels_batch
            )
            
            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss.item() < 20.0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                
                losses.append(total_loss.item())
                cox_losses.append(cox_loss.item() if isinstance(cox_loss, torch.Tensor) else 0.0)
                edl_losses.append(edl_loss.item() if isinstance(edl_loss, torch.Tensor) else 0.0)
                focal_losses.append(focal_loss.item() if isinstance(focal_loss, torch.Tensor) else 0.0)
           
        except Exception as e:
            print(f"Training error in batch {batch_idx}: {e}")
            continue
    
    return (np.mean(losses) if losses else 0, 
            np.mean(cox_losses) if cox_losses else 0, 
            np.mean(edl_losses) if edl_losses else 0,
            np.mean(focal_losses) if focal_losses else 0)

def evaluate(model, loader, device):
    model.eval()
    rna_risks, wsi_risks, fused_risks = [], [], []
    rna_probs, wsi_probs, fused_probs = [], [], []
    rna_uncertainties, wsi_uncertainties, fused_uncertainties = [], [], []
    labels, times = [], []
    
    with torch.no_grad():
        for batch_data in loader:
            try:
                if len(batch_data) == 5:
                    rna_batch, wsi_batch, clinical_batch, labels_batch, times_batch = batch_data
                    gene_expressions = rna_batch[0].to(device)
                    wsi_patches = wsi_batch[0].to(device)
                    clinical = clinical_batch[0].to(device)
                    label = labels_batch[0]
                    time = times_batch[0]
                else:
                    continue
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                outputs = model(gene_expressions, wsi_patches, clinical)
                
                # Extract risks
                rna_risks.append(outputs['rna_risk'].cpu().item())
                wsi_risks.append(outputs['wsi_risk'].cpu().item())
                fused_risks.append(outputs['fused_risk'].cpu().item())
                
                # Extract probabilities and uncertainties
                for alpha, probs_list, uncertainties_list in [
                    (outputs['rna_dirichlet'], rna_probs, rna_uncertainties),
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
    
    # Calculate metrics
    results = {}
    for prefix, risks_list, probs_list, uncertainties_list in [
        ('rna', rna_risks, rna_probs, rna_uncertainties),
        ('wsi', wsi_risks, wsi_probs, wsi_uncertainties),
        ('fused', fused_risks, fused_probs, fused_uncertainties)
    ]:
        if len(set(labels)) > 1 and len(risks_list) > 0:
            try:
                # C-index (primary metric)
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