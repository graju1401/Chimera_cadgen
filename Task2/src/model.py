import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import Mamba, fallback to transformer if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Mamba SSM available - using MambaFormer architecture")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using standard transformer")

class PatchEncoder(nn.Module):
    """WSI patch encoder"""
    def __init__(self, input_dim=1024, embed_dim=128, dropout=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, patch_features):
        return self.encoder(patch_features)

class ClinicalEncoder(nn.Module):
    """Clinical data encoder"""
    def __init__(self, input_dim, embed_dim=128, dropout=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.encoder(x)

class MambaTransformerLayer(nn.Module):
    """MambaFormer layer: Mamba for sequence modeling + Cross-attention with clinical features"""
    def __init__(self, d_model=128, d_state=16, d_conv=4, expand=2, n_heads=8, dropout=0.2):
        super().__init__()
        
        # Mamba component (if available)
        self.use_mamba = MAMBA_AVAILABLE
        if self.use_mamba:
            try:
                self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                self.mamba_norm = nn.LayerNorm(d_model)
            except Exception as e:
                print(f"Mamba initialization failed: {e}, falling back to transformer")
                self.use_mamba = False
        
        # Fallback: Standard transformer layer
        if not self.use_mamba:
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=d_model * 4,
                dropout=dropout, 
                batch_first=True, 
                activation='gelu'
            )
        
        # Cross-attention for clinical guidance
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embeddings, clinical_embedding):
        # Add batch dimension if needed
        if embeddings.dim() == 2:
            seq = embeddings.unsqueeze(0)
        else:
            seq = embeddings
        
        # 1. Mamba or Transformer for sequence modeling
        residual = seq
        if self.use_mamba:
            try:
                mamba_out = self.mamba(seq)
                seq = self.mamba_norm(mamba_out + residual)
            except Exception as e:
                print(f"Mamba forward error: {e}")
                # Fallback to identity
                seq = residual
        else:
            seq = self.transformer_layer(seq)
        
        seq = self.dropout(seq)
        
        # 2. Cross-attention with clinical features (clinical guides WSI)
        if clinical_embedding.dim() == 1:
            clinical_seq = clinical_embedding.view(1, 1, -1)
        elif clinical_embedding.dim() == 2:
            clinical_seq = clinical_embedding.unsqueeze(1)
        else:
            clinical_seq = clinical_embedding
        
        residual = seq
        cross_attn_out, attn_weights = self.cross_attention(seq, clinical_seq, clinical_seq)
        seq = self.cross_norm(cross_attn_out + residual)
        seq = self.dropout(seq)
        
        # 3. Feed-forward network
        residual = seq
        ffn_out = self.ffn(seq)
        seq = self.ffn_norm(ffn_out + residual)
        
        # Remove batch dimension if it was added
        updated_embeddings = seq.squeeze(0) if seq.dim() == 3 and seq.shape[0] == 1 else seq
        
        return updated_embeddings, attn_weights

class AttentionAggregator(nn.Module):
    """Attention-based aggregation from patches to patient level"""
    def __init__(self, d_model=128, n_heads=8, dropout=0.2):
        super().__init__()
        
        # Learnable query for patient-level representation
        self.patient_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention mechanism
        self.aggregation_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=n_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patch_embeddings):
        # Add batch dimension if needed
        if patch_embeddings.dim() == 2:
            patches = patch_embeddings.unsqueeze(0)
        else:
            patches = patch_embeddings
        
        # Use learnable query to aggregate patches
        query = self.patient_query.expand(patches.size(0), -1, -1)
        
        # Attention-based aggregation
        aggregated, attn_weights = self.aggregation_attention(
            query=query, 
            key=patches, 
            value=patches
        )
        
        aggregated = self.norm(aggregated)
        aggregated = self.dropout(aggregated)
        
        # Remove query dimension
        patient_embedding = aggregated.squeeze(1) if aggregated.size(1) == 1 else aggregated
        
        return patient_embedding, attn_weights

class WSIMambaFormer(nn.Module):
    """WSI + Clinical MambaFormer for patient-level representation"""
    def __init__(self, wsi_dim, clinical_dim, d_model=128, n_layers=3, d_state=16, 
                 d_conv=4, expand=2, n_heads=8, dropout=0.2, max_patches=1500):
        super().__init__()
        self.max_patches = max_patches
        
        # Encoders
        self.wsi_encoder = PatchEncoder(wsi_dim, d_model, dropout)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, d_model, dropout)
        
        # MambaFormer layers
        self.mambaformer_layers = nn.ModuleList([
            MambaTransformerLayer(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv,
                expand=expand, 
                n_heads=n_heads, 
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Attention-based aggregation
        self.aggregator = AttentionAggregator(d_model=d_model, n_heads=n_heads, dropout=dropout)
        
        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, patch_features, clinical_features):
        # Limit patch count if necessary
        if patch_features.shape[0] > self.max_patches:
            indices = torch.randperm(patch_features.shape[0], device=patch_features.device)[:self.max_patches]
            patch_features = patch_features[indices]
        
        # 1. Encode patches and clinical data
        wsi_patch_embeddings = self.wsi_encoder(patch_features)
        clinical_embedding = self.clinical_encoder(clinical_features)
        
        # 2. Apply MambaFormer layers (clinical guides WSI processing)
        current_embeddings = wsi_patch_embeddings
        all_attention_weights = []
        
        for layer in self.mambaformer_layers:
            current_embeddings, attn_weights = layer(current_embeddings, clinical_embedding)
            if attn_weights is not None:
                all_attention_weights.append(attn_weights)
        
        # 3. Attention-based aggregation to patient level
        patient_embedding, aggregation_weights = self.aggregator(current_embeddings)
        all_attention_weights.append(aggregation_weights)
        
        # 4. Final projection
        final_embedding = self.out_proj(patient_embedding)
        
        return final_embedding, all_attention_weights

class WSIEDLModel(nn.Module):
    """WSI + Clinical EDL Model with MambaFormer"""
    def __init__(self, wsi_dim, clinical_dim, config):
        super().__init__()
        
        # MambaFormer backbone
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
            max_patches=config['model']['max_patches_attention']
        )
        
        # EDL classifier head
        d_model = config['model']['d_model']
        dropout = config['model']['dropout']
        
        self.edl_classifier = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)  # Raw outputs for EDL (will be converted to evidence)
        )

    def forward(self, patch_features, clinical_features):
        # Get patient-level embedding from MambaFormer
        fused_embedding, attention_weights = self.mambaformer(patch_features, clinical_features)
        
        if fused_embedding.dim() == 1:
            fused_embedding = fused_embedding.unsqueeze(0)
        
        # Get EDL predictions (raw logits that will be converted to evidence)
        pred_logits = self.edl_classifier(fused_embedding)
        
        return {
            'pred_logits': pred_logits,
            'fused_embedding': fused_embedding,
            'attention_weights': attention_weights
        }