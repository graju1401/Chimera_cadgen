import torch
import torch.nn as nn
import torch.nn.functional as F
from pycox.models.loss import CoxPHLoss
from edl_pytorch import evidential_classification

class FocalLoss(nn.Module):
    """Improved Focal Loss implementation"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for single modality (MRI or WSI) + Clinical"""
    
    def __init__(self, cox_weight=1.5, edl_weight=0.3, lamb=0.0005):
        super().__init__()
        self.cox = CoxPHLoss()
        self.cox_weight = cox_weight
        self.edl_weight = edl_weight
        self.lamb = lamb

    def forward(self, fused_risk, pred_dirichlet, durations, events, labels):
        device = fused_risk.device
        
        # Cox loss
        if len(fused_risk) > 1 and torch.sum(events) > 0:
            try:
                if torch.var(fused_risk) > 1e-6:
                    cox_loss = self.cox(fused_risk.view(-1), durations.view(-1), events.view(-1))
                    cox_loss = torch.clamp(cox_loss, 0.0, 5.0)
                else:
                    cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
            except:
                cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # EDL classification loss
        try:
            if pred_dirichlet.dim() == 3:
                pred_dirichlet_clean = pred_dirichlet.view(-1, pred_dirichlet.shape[-1])
            else:
                pred_dirichlet_clean = pred_dirichlet
            pred_dirichlet_clean = torch.clamp(pred_dirichlet_clean, min=1.001, max=50.0)
            edl_loss = evidential_classification(pred_dirichlet_clean, labels.long().unsqueeze(1), lamb=self.lamb)
            edl_loss = torch.clamp(edl_loss, 0.0, 5.0)
        except:
            if pred_dirichlet.dim() == 3:
                pred_dirichlet_clean = pred_dirichlet.view(-1, pred_dirichlet.shape[-1])
            else:
                pred_dirichlet_clean = pred_dirichlet
            probs = F.softmax(pred_dirichlet_clean, dim=1)
            edl_loss = F.cross_entropy(probs, labels.long())
        
        total_loss = self.cox_weight * cox_loss + self.edl_weight * edl_loss
        
        return total_loss, cox_loss, edl_loss


class MultiModalCombinedLoss(nn.Module):
    """Combined loss for multi-modal fusion"""
    
    def __init__(self, cox_weight=1.0, edl_weight=0.5, focal_weight=0.4, lamb=0.001):
        super().__init__()
        self.cox = CoxPHLoss()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.cox_weight = cox_weight
        self.edl_weight = edl_weight
        self.focal_weight = focal_weight
        self.lamb = lamb

    def forward(self, mri_risk, wsi_risk, fused_risk, mri_dirichlet, wsi_dirichlet, 
                fused_dirichlet, durations, events, labels):
        
        device = mri_risk.device
        
        # Cox loss for survival prediction
        if len(fused_risk) > 1 and torch.sum(events) > 0:
            try:
                if torch.var(fused_risk) > 1e-6:
                    cox_loss = self.cox(fused_risk.view(-1), durations.view(-1), events.view(-1))
                    cox_loss = torch.clamp(cox_loss, 0.0, 3.0)
                else:
                    cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
            except:
                cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            cox_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # EDL loss for uncertainty quantification
        try:
            if fused_dirichlet.dim() == 3:
                fused_dirichlet_clean = fused_dirichlet.view(-1, fused_dirichlet.shape[-1])
            else:
                fused_dirichlet_clean = fused_dirichlet
            fused_dirichlet_clean = torch.clamp(fused_dirichlet_clean, min=1.001, max=50.0)
            edl_loss = evidential_classification(fused_dirichlet_clean, labels.long().unsqueeze(1), lamb=self.lamb)
            edl_loss = torch.clamp(edl_loss, 0.0, 3.0)
        except:
            if fused_dirichlet.dim() == 3:
                fused_dirichlet_clean = fused_dirichlet.view(-1, fused_dirichlet.shape[-1])
            else:
                fused_dirichlet_clean = fused_dirichlet
            probs = F.softmax(fused_dirichlet_clean, dim=1)
            edl_loss = F.cross_entropy(probs, labels.long())
        
        # Focal loss
        try:
            if fused_dirichlet.dim() == 3:
                fused_dirichlet_clean = fused_dirichlet.view(-1, fused_dirichlet.shape[-1])
            else:
                fused_dirichlet_clean = fused_dirichlet
            
            focal_loss = self.focal(fused_dirichlet_clean, labels.long())
            
        except Exception as e:
            focal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Combine losses
        total_loss = (self.cox_weight * cox_loss + 
                     self.edl_weight * edl_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss, cox_loss, edl_loss, focal_loss

