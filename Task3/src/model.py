import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from edl_pytorch import Dirichlet, NormalInvGamma
from losses import DS_Combin_two

class RNAGeneEncoder(nn.Module):
    def __init__(self, input_dim=1, embed_dim=128, n_layers=2, d_state=16, d_conv=4, expand=2, dropout=0.4):
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

    def forward(self, gene_expressions):
        # gene_expressions: (num_genes, 1)
        batch_size = 500
        num_genes = gene_expressions.shape[0]
        all_embeddings = []
        
        for i in range(0, num_genes, batch_size):
            batch_genes = gene_expressions[i:i+batch_size]
            
            h = self.input_proj(batch_genes)
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
        
        gene_embeddings = torch.cat(all_embeddings, dim=0)
        return gene_embeddings

class ClinicalEncoder(nn.Module):
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
        
        # Cross-attention with clinical embeddings as key/value
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

class RNAMambaFormer(nn.Module):
    def __init__(self, clinical_dim, d_model=128, n_layers=3, d_state=16, 
                 d_conv=4, expand=2, n_heads=4, dropout=0.4, max_genes_attention=1000):
        super().__init__()
        self.max_genes_attention = max_genes_attention
        
        self.rna_encoder = RNAGeneEncoder(
            input_dim=1, embed_dim=d_model, n_layers=2, d_state=d_state, 
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

    def forward(self, gene_expressions, clinical_features):
        # gene_expressions: (num_genes, 1)
        rna_gene_embeddings = self.rna_encoder(gene_expressions)
        clinical_embedding = self.clinical_encoder(clinical_features)
        
        num_genes = rna_gene_embeddings.shape[0]
        if num_genes > self.max_genes_attention:
            indices = torch.randperm(num_genes)[:self.max_genes_attention]
            rna_gene_embeddings = rna_gene_embeddings[indices]
        
        current_embeddings = rna_gene_embeddings
        
        # Apply MambaFormer layers with clinical cross-attention
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

class MambaPatchEncoder(nn.Module):
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

class WSIMambaFormer(nn.Module):
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
        
        # Apply MambaFormer layers with clinical cross-attention
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

class MultiModalEvidentialModel(nn.Module):
    def __init__(self, wsi_dim, clinical_dim, config):
        super().__init__()
        
        # Extract configuration
        mambaformer_config = config['model']['mambaformer']
        dropout = mambaformer_config['dropout']
        
        # Individual modality models (RNA replaces MRI)
        self.rna_mambaformer = RNAMambaFormer(
            clinical_dim, 
            d_model=mambaformer_config['d_model'],
            n_layers=mambaformer_config['n_layers'],
            d_state=mambaformer_config['d_state'],
            d_conv=mambaformer_config['d_conv'],
            expand=mambaformer_config['expand'],
            n_heads=mambaformer_config['n_heads'],
            dropout=dropout,
            max_genes_attention=config['data']['max_genes_attention']
        )
        
        self.wsi_mambaformer = WSIMambaFormer(
            wsi_dim, 
            clinical_dim,
            d_model=mambaformer_config['d_model'],
            n_layers=mambaformer_config['n_layers'],
            d_state=mambaformer_config['d_state'],
            d_conv=mambaformer_config['d_conv'],
            expand=mambaformer_config['expand'],
            n_heads=mambaformer_config['n_heads'],
            dropout=dropout,
            max_patches_attention=config['data']['max_patches_attention']
        )
        
        # EDL heads for each modality
        self.rna_classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(dropout),
            Dirichlet(16, 2)
        )
        
        self.rna_regressor = nn.Sequential(
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

    def forward(self, gene_expressions, wsi_patches, clinical_features):
        # Process each modality (RNA and WSI both use clinical as K/V in cross-attention)
        rna_embedding = self.rna_mambaformer(gene_expressions, clinical_features)
        wsi_embedding = self.wsi_mambaformer(wsi_patches, clinical_features)
        
        # Ensure correct dimensions
        if rna_embedding.dim() == 1:
            rna_embedding = rna_embedding.unsqueeze(0)
        if wsi_embedding.dim() == 1:
            wsi_embedding = wsi_embedding.unsqueeze(0)
        
        # Individual predictions
        rna_dirichlet = self.rna_classifier(rna_embedding)
        rna_nig = self.rna_regressor(rna_embedding)
        rna_risk = rna_nig[0].squeeze(-1)
        
        wsi_dirichlet = self.wsi_classifier(wsi_embedding)
        wsi_nig = self.wsi_regressor(wsi_embedding)
        wsi_risk = wsi_nig[0].squeeze(-1)
        
        # Dempster-Shafer fusion of Dirichlet distributions
        fused_dirichlet = DS_Combin_two(rna_dirichlet, wsi_dirichlet, n_classes=2)
        
        # Combined embedding for risk prediction
        combined_embedding = torch.cat([rna_embedding, wsi_embedding], dim=-1)
        fused_nig = self.fusion_regressor(combined_embedding)
        fused_risk = fused_nig[0].squeeze(-1)
        
        return {
            'rna_risk': rna_risk,
            'wsi_risk': wsi_risk,
            'fused_risk': fused_risk,
            'rna_dirichlet': rna_dirichlet,
            'wsi_dirichlet': wsi_dirichlet,
            'fused_dirichlet': fused_dirichlet,
            'rna_embedding': rna_embedding,
            'wsi_embedding': wsi_embedding
        }