import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
from losses import softplus_evidence

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    losses = []
    
    # Set epoch for criterion annealing
    if hasattr(criterion, 'set_epoch'):
        criterion.set_epoch(epoch)
    
    optimizer.zero_grad()
    
    for batch_idx, batch_data in enumerate(loader):
        try:
            patches_batch, clinical_batch, labels_batch = batch_data
            batch_size = len(patches_batch)
            all_outputs = []
            
            # Process each sample in the batch
            for i in range(batch_size):
                patches = patches_batch[i].to(device, non_blocking=True)
                clinical = clinical_batch[i].to(device, non_blocking=True)
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                outputs = model(patches, clinical)
                all_outputs.append(outputs)
            
            # Combine outputs from batch
            pred_logits_list = []
            for out in all_outputs:
                logits = out['pred_logits']
                if logits.dim() > 2:
                    logits = logits.squeeze(0)
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                pred_logits_list.append(logits)
            
            pred_logits = torch.cat(pred_logits_list, dim=0)
            labels_batch = labels_batch.to(device, non_blocking=True)
            
            # Calculate loss
            loss = criterion(pred_logits, labels_batch)
            
            # Check for valid loss
            if not torch.isnan(loss) and not torch.isinf(loss) and loss.item() < 50.0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                losses.append(loss.item())
            else:
                print(f"Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
            
        except Exception as e:
            print(f"Training batch {batch_idx} error: {e}")
            continue
    
    return np.mean(losses) if losses else 0.0

def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold for binary classification"""
    if len(set(y_true)) < 2:
        return 0.5, 0.0
    
    try:
        # Use precision-recall curve to find optimal F1 threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        if len(precision) == 0 or len(recall) == 0:
            return 0.5, 0.0
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        if len(f1_scores) > 0 and len(thresholds) > 0:
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
            optimal_f1 = f1_scores[optimal_idx]
        else:
            optimal_threshold, optimal_f1 = 0.5, 0.0
            
    except Exception as e:
        print(f"Error in threshold optimization: {e}")
        optimal_threshold, optimal_f1 = 0.5, 0.0
    
    return optimal_threshold, optimal_f1

def evaluate(model, loader, device):
    """Evaluate model and return metrics with uncertainty"""
    model.eval()
    probs, labels, uncertainties = [], [], []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            try:
                patches_batch, clinical_batch, labels_batch = batch_data
                
                patches = patches_batch[0].to(device, non_blocking=True)
                clinical = clinical_batch[0].to(device, non_blocking=True)
                label = labels_batch[0]
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                outputs = model(patches, clinical)
                
                # Get predictions and uncertainty from EDL
                raw_output = outputs['pred_logits']
                if raw_output.dim() > 2:
                    raw_output = raw_output.squeeze(0)
                elif raw_output.dim() == 1:
                    raw_output = raw_output.unsqueeze(0)
                
                # Convert to evidence and compute uncertainty
                evidence = softplus_evidence(raw_output)
                alpha = evidence + 1
                S = alpha.sum(dim=1)
                
                # Probability for class 1 (BRS3)
                prob = (alpha[:, 1] / S).cpu().item()
                
                # Uncertainty calculation
                uncertainty = (2.0 / S).cpu().item()
                
                probs.append(prob)
                uncertainties.append(uncertainty)
                labels.append(int(label.item()))
                
                # Store attention weights for visualization
                if 'attention_weights' in outputs and outputs['attention_weights']:
                    all_attention_weights.append(outputs['attention_weights'])
                
            except Exception as e:
                print(f"Evaluation batch {batch_idx} error: {e}")
                continue
    
    # Calculate metrics
    if len(set(labels)) > 1 and len(probs) > 0:
        try:
            # Calculate AUC
            auc = roc_auc_score(labels, probs)
            
            # Find optimal threshold and F1
            optimal_threshold, optimal_f1 = find_optimal_threshold(labels, probs)
            
            # Calculate accuracy using optimal threshold
            pred_labels = (np.array(probs) > optimal_threshold).astype(int)
            accuracy = accuracy_score(labels, pred_labels)
            mean_uncertainty = np.mean(uncertainties) if uncertainties else 0.5
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            auc = accuracy = optimal_f1 = optimal_threshold = 0.5
            mean_uncertainty = 0.5
    else:
        auc = accuracy = optimal_f1 = optimal_threshold = 0.5
        mean_uncertainty = 0.5
        print(f"Cannot calculate metrics - only {len(set(labels))} unique label(s) in {len(labels)} samples")
    
    return {
        'f1_score': optimal_f1,
        'auc': auc, 
        'accuracy': accuracy,
        'uncertainty': mean_uncertainty,
        'threshold': optimal_threshold,
        'n_samples': len(labels),
        'predictions': probs,
        'true_labels': labels,
        'uncertainties': uncertainties,
        'attention_weights': all_attention_weights
    }

def run_training(model, train_loader, test_loader, optimizer, criterion, scheduler, config, device):
    """Main training loop"""
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    best_f1 = 0
    best_auc = 0
    best_combined_metric = 0
    best_epoch = 0
    history = []
    patience_counter = 0
    
    print(f"Training for {epochs} epochs with patience {patience}")
    print(f"Loss function: {type(criterion).__name__}")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validation
        val_results = evaluate(model, test_loader, device)
        
        # Combined metric for best model selection (F1 + AUC)
        combined_metric = val_results['f1_score'] + val_results['auc']
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(combined_metric)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if this is the best model
        if combined_metric > best_combined_metric:
            best_combined_metric = combined_metric
            best_f1 = val_results['f1_score']
            best_auc = val_results['auc']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save model state
            if config['output']['save_model']:
                try:
                    torch.save(model.state_dict(), config['output']['model_save_path'])
                    print(f"🎯 New best model saved! F1={best_f1:.4f}, AUC={best_auc:.4f}")
                except Exception as e:
                    print(f"Error saving model: {e}")
        else:
            patience_counter += 1
        
        # Store epoch results
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'learning_rate': current_lr,
            'combined_metric': combined_metric,
            **{k: v for k, v in val_results.items() if k not in ['predictions', 'true_labels', 'uncertainties', 'attention_weights']}
        }
        history.append(epoch_data)
        
        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, F1={val_results['f1_score']:.3f}, "
              f"AUC={val_results['auc']:.3f}, Unc={val_results['uncertainty']:.3f}, "
              f"Combined={combined_metric:.3f}")
        print(f"LR={current_lr:.2e}, Patience={patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
        
        if current_lr <= config['scheduler']['min_lr']:
            print("Minimum learning rate reached")
            break
    
    print(f"\n🏆 Training Summary:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best F1-score: {best_f1:.4f}")
    print(f"   Best AUC: {best_auc:.4f}")
    print(f"   Best combined metric: {best_combined_metric:.4f}")
    
    # Load best model if saved
    if config['output']['save_model'] and best_epoch > 0:
        try:
            model.load_state_dict(torch.load(config['output']['model_save_path'], map_location=device))
            print(f"Loaded best model from epoch {best_epoch}")
        except Exception as e:
            print(f"Error loading best model: {e}")
    
    return history, best_combined_metric, best_epoch