import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from edl_pytorch import Dirichlet, NormalInvGamma



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


# Encoder Components

class MRISliceEncoder(nn.Module):
    """MRI slice encoder using Mamba"""
    
    def __init__(self, input_dim=100, embed_dim=128, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, slice_features):
        batch_size = 500
        num_slices = slice_features.shape[0]
        all_embeddings = []
        
        for i in range(0, num_slices, batch_size):
            batch_slices = slice_features[i:i+batch_size]
            
            h = self.input_proj(batch_slices)
            h = self.input_norm(h)
            h = self.input_dropout(h)
            h = h.unsqueeze(0)
            
            for mamba, norm, dropout in zip(self.mamba_layers, self.norms, self.dropouts):
                residual = h
                h = mamba(h)
                h = norm(h + residual)
                h = dropout(h)
            
            h = h.squeeze(0)
            h = self.output_proj(h)
            all_embeddings.append(h.detach())
            
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        slice_embeddings = torch.cat(all_embeddings, dim=0)
        return slice_embeddings


class MambaPatchEncoder(nn.Module):
    """WSI patch encoder using Mamba"""
    
    def __init__(self, input_dim=1024, embed_dim=128, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, patch_features):
        batch_size = 1000
        num_patches = patch_features.shape[0]
        all_embeddings = []
        
        for i in range(0, num_patches, batch_size):
            batch_patches = patch_features[i:i+batch_size]
            
            h = self.input_proj(batch_patches)
            h = self.input_norm(h)
            h = self.input_dropout(h)
            h = h.unsqueeze(0)
            
            for mamba, norm, dropout in zip(self.mamba_layers, self.norms, self.dropouts):
                residual = h
                h = mamba(h)
                h = norm(h + residual)
                h = dropout(h)
            
            h = h.squeeze(0)
            h = self.output_proj(h)
            all_embeddings.append(h.detach())
            
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
        
        patch_embeddings = torch.cat(all_embeddings, dim=0)
        return patch_embeddings


class ClinicalEncoder(nn.Module):
    """Clinical data encoder"""
    
    def __init__(self, input_dim, embed_dim=128, dropout=0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.encoder(x)


class MambaFormerCrossAttentionLayer(nn.Module):
    """Cross-attention layer combining Mamba and Transformer attention"""
    
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, n_heads=4, dropout=0.4):
        super().__init__()
        
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_norm = nn.LayerNorm(d_model)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, embeddings, clinical_embedding):
        seq = embeddings.unsqueeze(0)
        
        # Mamba branch
        residual = seq
        mamba_out = self.mamba(seq)
        seq = self.mamba_norm(mamba_out + residual)
        seq = self.dropout1(seq)
        
        # Cross-attention
        clinical_seq = clinical_embedding.view(1, 1, -1)
        
        residual = seq
        cross_attended, cross_attn_weights = self.cross_attention(
            query=seq, key=clinical_seq, value=clinical_seq
        )
        seq = self.cross_attention_norm(cross_attended + residual)
        seq = self.dropout2(seq)
        
        # FFN
        residual = seq
        ffn_out = self.ffn(seq)
        seq = self.ffn_norm(ffn_out + residual)
        
        updated_embeddings = seq.squeeze(0)
        cross_attention_weights = cross_attn_weights.mean(dim=1).squeeze(0)
        
        return updated_embeddings, cross_attention_weights



# Main Model Architectures

class MRIMambaFormer(nn.Module):
    """MRI + Clinical MambaFormer"""
    
    def __init__(self, mri_dim, clinical_dim, d_model=128, n_layers=3, d_state=16, 
                 d_conv=4, expand=2, n_heads=4, dropout=0.4, max_slices_attention=1000):
        super().__init__()
        self.max_slices_attention = max_slices_attention
        
        self.mri_encoder = MRISliceEncoder(
            mri_dim, d_model, n_layers=2, d_state=d_state, 
            d_conv=d_conv, expand=expand, dropout=dropout
        )
        self.clinical_encoder = ClinicalEncoder(clinical_dim, d_model, dropout)
        
        self.mambaformer_layers = nn.ModuleList([
            MambaFormerCrossAttentionLayer(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, n_heads=n_heads, dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.aggregation_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, slice_features, clinical_features):
        mri_slice_embeddings = self.mri_encoder(slice_features)
        clinical_embedding = self.clinical_encoder(clinical_features)
        
        num_slices = mri_slice_embeddings.shape[0]
        if num_slices > self.max_slices_attention:
            indices = torch.randperm(num_slices)[:self.max_slices_attention]
            mri_slice_embeddings = mri_slice_embeddings[indices]
        
        current_embeddings = mri_slice_embeddings
        
        for layer in self.mambaformer_layers:
            current_embeddings, _ = layer(current_embeddings, clinical_embedding)
        
        seq = current_embeddings.unsqueeze(0)
        aggregated, final_attn_weights = self.aggregation_attention(
            query=seq, key=seq, value=seq
        )
        
        attn_scores = final_attn_weights.mean(1).squeeze(0)
        embedding = torch.sum(
            attn_scores.unsqueeze(-1) * aggregated.squeeze(0), dim=0
        )
        
        final_embedding = self.out_proj(embedding)
        return final_embedding


class WSIMambaFormer(nn.Module):
    """WSI + Clinical MambaFormer"""
    
    def __init__(self, wsi_dim, clinical_dim, d_model=128, n_layers=3, d_state=16, 
                 d_conv=4, expand=2, n_heads=4, dropout=0.4, max_patches_attention=5000):
        super().__init__()
        self.max_patches_attention = max_patches_attention
        
        self.wsi_encoder = MambaPatchEncoder(
            wsi_dim, d_model, n_layers=2, d_state=d_state, 
            d_conv=d_conv, expand=expand, dropout=dropout
        )
        self.clinical_encoder = ClinicalEncoder(clinical_dim, d_model, dropout)
        
        self.mambaformer_layers = nn.ModuleList([
            MambaFormerCrossAttentionLayer(
                d_model=d_model, d_state=d_state, d_conv=d_conv,
                expand=expand, n_heads=n_heads, dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.aggregation_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, patch_features, clinical_features):
        wsi_patch_embeddings = self.wsi_encoder(patch_features)
        clinical_embedding = self.clinical_encoder(clinical_features)
        
        num_patches = wsi_patch_embeddings.shape[0]
        if num_patches > self.max_patches_attention:
            indices = torch.randperm(num_patches)[:self.max_patches_attention]
            wsi_patch_embeddings = wsi_patch_embeddings[indices]
        
        current_embeddings = wsi_patch_embeddings
        
        for layer in self.mambaformer_layers:
            current_embeddings, _ = layer(current_embeddings, clinical_embedding)
        
        seq = current_embeddings.unsqueeze(0)
        aggregated, final_attn_weights = self.aggregation_attention(
            query=seq, key=seq, value=seq
        )
        
        attn_scores = final_attn_weights.mean(1).squeeze(0)
        embedding = torch.sum(
            attn_scores.unsqueeze(-1) * aggregated.squeeze(0), dim=0
        )
        
        final_embedding = self.out_proj(embedding)
        return final_embedding


# Final Models

class MRIEDLModel(nn.Module):
    """MRI + Clinical EDL Model"""
    
    def __init__(self, mri_dim, clinical_dim, config):
        super().__init__()
        
        self.mambaformer = MRIMambaFormer(
            mri_dim=mri_dim,
            clinical_dim=clinical_dim,
            d_model=config['model']['d_model'],
            n_layers=config['model']['n_layers'],
            d_state=config['model']['d_state'],
            d_conv=config['model']['d_conv'],
            expand=config['model']['expand'],
            n_heads=config['model']['n_heads'],
            dropout=config['model']['dropout'],
            max_slices_attention=config['model']['max_slices_attention']
        )
        
        dropout = config['model']['dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            Dirichlet(16, 2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            NormalInvGamma(16, 1)
        )

    def forward(self, slice_features, clinical_features):
        fused_embedding = self.mambaformer(slice_features, clinical_features)
        
        if fused_embedding.dim() == 1:
            fused_embedding = fused_embedding.unsqueeze(0)
        
        pred_dirichlet = self.classifier(fused_embedding)
        pred_nig = self.regressor(fused_embedding)
        fused_risk = pred_nig[0].squeeze(-1)
        
        return {
            'fused_risk': fused_risk,
            'pred_dirichlet': pred_dirichlet,
            'pred_nig': pred_nig,
            'fused_embedding': fused_embedding
        }


class WSIEDLModel(nn.Module):
    """WSI + Clinical EDL Model"""
    
    def __init__(self, wsi_dim, clinical_dim, config):
        super().__init__()
        
        self.mambaformer = WSIMambaFormer(
            wsi_dim=wsi_dim,
            clinical_dim=clinical_dim,
            d_model=config['model']['d_model'],
            n_layers=config['model']['n_layers'],
            d_state=config['model']['d_state'],
            d_conv=config['model']['d_conv'],
            expand=config['model']['expand'],
            n_heads=config['model']['n_heads'],
            dropout=config['model']['dropout'],
            max_patches_attention=config['model']['max_patches_attention']
        )
        
        dropout = config['model']['dropout']
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            Dirichlet(16, 2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            NormalInvGamma(16, 1)
        )

    def forward(self, patch_features, clinical_features):
        fused_embedding = self.mambaformer(patch_features, clinical_features)
        
        if fused_embedding.dim() == 1:
            fused_embedding = fused_embedding.unsqueeze(0)
        
        pred_dirichlet = self.classifier(fused_embedding)
        pred_nig = self.regressor(fused_embedding)
        fused_risk = pred_nig[0].squeeze(-1)
        
        return {
            'fused_risk': fused_risk,
            'pred_dirichlet': pred_dirichlet,
            'pred_nig': pred_nig,
            'fused_embedding': fused_embedding
        }


class MultiModalEvidentialModel(nn.Module):
    """Multi-modal MRI + WSI + Clinical EDL Model"""
    
    def __init__(self, mri_dim, wsi_dim, clinical_dim, config):
        super().__init__()
        
        # Individual modality models
        self.mri_mambaformer = MRIMambaFormer(
            mri_dim=mri_dim,
            clinical_dim=clinical_dim,
            d_model=config['model']['d_model'],
            n_layers=config['model']['n_layers'],
            d_state=config['model']['d_state'],
            d_conv=config['model']['d_conv'],
            expand=config['model']['expand'],
            n_heads=config['model']['n_heads'],
            dropout=config['model']['dropout'],
            max_slices_attention=config['model']['max_slices_attention']
        )
        
        self.wsi_mambaformer = WSIMambaFormer(
            wsi_dim=wsi_dim,
            clinical_dim=clinical_dim,
            d_model=config['model']['d_model'],
            n_layers=config['model']['n_layers'],
            d_state=config['model']['d_state'],
            d_conv=config['model']['d_conv'],
            expand=config['model']['expand'],
            n_heads=config['model']['n_heads'],
            dropout=config['model']['dropout'],
            max_patches_attention=config['model']['max_patches_attention']
        )
        
        dropout = config['model']['dropout']
        
        # EDL heads for each modality
        self.mri_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            Dirichlet(16, 2)
        )
        
        self.mri_regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            NormalInvGamma(16, 1)
        )
        
        self.wsi_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            Dirichlet(16, 2)
        )
        
        self.wsi_regressor = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            NormalInvGamma(16, 1)
        )
        
        # Fusion regressor for combined risk
        self.fusion_regressor = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            NormalInvGamma(32, 1)
        )

    def forward(self, mri_slices, wsi_patches, clinical_features):
        # Process each modality
        mri_embedding = self.mri_mambaformer(mri_slices, clinical_features)
        wsi_embedding = self.wsi_mambaformer(wsi_patches, clinical_features)
        
        # Ensure correct dimensions
        if mri_embedding.dim() == 1:
            mri_embedding = mri_embedding.unsqueeze(0)
        if wsi_embedding.dim() == 1:
            wsi_embedding = wsi_embedding.unsqueeze(0)
        
        # Individual predictions
        mri_dirichlet = self.mri_classifier(mri_embedding)
        mri_nig = self.mri_regressor(mri_embedding)
        mri_risk = mri_nig[0].squeeze(-1)
        
        wsi_dirichlet = self.wsi_classifier(wsi_embedding)
        wsi_nig = self.wsi_regressor(wsi_embedding)
        wsi_risk = wsi_nig[0].squeeze(-1)
        
        # Dempster-Shafer fusion of Dirichlet distributions
        fused_dirichlet = DS_Combin_two(mri_dirichlet, wsi_dirichlet, n_classes=2)
        
        # Combined embedding for risk prediction
        combined_embedding = torch.cat([mri_embedding, wsi_embedding], dim=-1)
        fused_nig = self.fusion_regressor(combined_embedding)
        fused_risk = fused_nig[0].squeeze(-1)
        
        return {
            'mri_risk': mri_risk,
            'wsi_risk': wsi_risk,
            'fused_risk': fused_risk,
            'mri_dirichlet': mri_dirichlet,
            'wsi_dirichlet': wsi_dirichlet,
            'fused_dirichlet': fused_dirichlet,
            'mri_embedding': mri_embedding,
            'wsi_embedding': wsi_embedding
        }