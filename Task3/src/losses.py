import torch
import torch.nn as nn
import torch.nn.functional as F
from pycox.models.loss import CoxPHLoss
from edl_pytorch import evidential_classification

def DS_Combin_two(alpha1, alpha2, n_classes=2):
    """Dempster-Shafer combination of two Dirichlet distributions"""
    eps = 1e-8
    alpha1 = torch.clamp(alpha1, min=1.0 + eps)
    alpha2 = torch.clamp(alpha2, min=1.0 + eps)
    
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape) + eps)
        u[v] = n_classes / (S[v] + eps)

    bb = torch.bmm(b[0].view(-1, n_classes, 1), b[1].view(-1, 1, n_classes))
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    K = bb_sum - bb_diag
    K = torch.clamp(K, max=0.99)

    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape) + eps)
    u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape) + eps)
    S_a = n_classes / (u_a + eps)
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    alpha_a = torch.clamp(alpha_a, min=1.0 + eps, max=100.0)
    return alpha_a

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class MultiModalCombinedLoss(nn.Module):
    def __init__(self, cox_weight=1.0, edl_weight=0.5, focal_weight=0.4, lamb=0.001, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cox = CoxPHLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.cox_weight = cox_weight
        self.edl_weight = edl_weight
        self.focal_weight = focal_weight
        self.lamb = lamb

    def forward(self, rna_risk, wsi_risk, fused_risk, rna_dirichlet, wsi_dirichlet, 
                fused_dirichlet, durations, events, labels):
        
        device = rna_risk.device
        
        # Cox loss for survival prediction (time to HG recurrence)
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
            print(f"Focal loss error: {e}")
            focal_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Combine losses
        total_loss = (self.cox_weight * cox_loss + 
                     self.edl_weight * edl_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss, cox_loss, edl_loss, focal_loss