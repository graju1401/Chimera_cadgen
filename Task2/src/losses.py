import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def softplus_evidence(y):
    """Convert raw outputs to evidence using softplus"""
    return F.softplus(y)

def kl_divergence(alpha, num_classes, device=None):
    """KL divergence for EDL regularization"""
    if device is None:
        device = alpha.device
    
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    first_term = (
        torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True) +
        torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha))
    ).sum(dim=1, keepdim=True)
    
    kl = first_term + second_term
    return kl

def compute_class_weights(cls_num_list, beta=0.95):
    """Compute simple class weights for imbalanced data"""
    effective_num = 1.0 - np.power(beta, cls_num_list)
    weights = (1.0 - beta) / effective_num
    weights = weights / np.sum(weights) * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)

class EDLLoss(nn.Module):
    """Simplified Evidential Deep Learning Loss"""
    def __init__(self, num_classes=2, annealing_step=5, uncertainty_reg=0.01, 
                 cls_num_list=None, beta=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.uncertainty_reg = uncertainty_reg
        self.current_epoch = 0
        
        # Class balance weights
        if cls_num_list is not None:
            self.class_weights = compute_class_weights(cls_num_list, beta)
        else:
            self.class_weights = None

    def set_epoch(self, epoch):
        """Update current epoch for annealing"""
        self.current_epoch = epoch

    def forward(self, logits, target):
        device = logits.device
        
        # Convert logits to evidence using softplus
        evidence = softplus_evidence(logits)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # One-hot encode targets
        target_onehot = F.one_hot(target.long(), self.num_classes).float().to(device)
        
        # Main EDL loss term
        A = torch.sum(target_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # Annealing coefficient (gradually increase KL weight)
        annealing_coef = min(1.0, self.current_epoch / self.annealing_step)
        
        # KL divergence for uncertainty regularization
        kl_alpha = (alpha - 1) * (1 - target_onehot) + 1
        kl_div = kl_divergence(kl_alpha, self.num_classes, device).squeeze()
        
        # Uncertainty regularization (penalize high uncertainty)
        uncertainty = self.num_classes / S.squeeze()
        uncertainty_loss = self.uncertainty_reg * torch.mean(uncertainty)
        
        # Main EDL loss
        edl_loss = A + annealing_coef * kl_div
        
        # Apply class weights if available
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)
            sample_weights = torch.sum(class_weights * target_onehot, dim=1)
            edl_loss = sample_weights * edl_loss
        
        # Total loss = EDL loss + uncertainty regularization
        total_loss = edl_loss + uncertainty_loss
        
        return total_loss.mean()

class FocalLoss(nn.Module):
    """Focal Loss for comparison (optional)"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss