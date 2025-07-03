import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler

def custom_collate_fn_wsi(batch):
    """Custom collate function for WSI data"""
    patches_batch = []
    clinical_batch = []
    labels_batch = []
    
    for patches, clinical, label in batch:
        patches_batch.append(patches)
        clinical_batch.append(clinical)
        labels_batch.append(label)
    
    return patches_batch, clinical_batch, torch.stack(labels_batch)

class WSIPatchAugmentation:
    """Essential data augmentation for WSI patches"""
    def __init__(self, noise_std=0.02, dropout_rate=0.15, training=True):
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate
        self.training = training
    
    def __call__(self, patches, is_minority_class=False):
        if not self.training:
            return patches
            
        # Applying stronger augmentation for minority class to address class-imbalance
        augmentation_factor = 1.5 if is_minority_class else 1.0
        
        # 1. Gaussian noise 
        if torch.rand(1) < 0.6 * augmentation_factor:
            noise = torch.randn_like(patches) * (self.noise_std * augmentation_factor)
            patches = patches + noise
        
        # 2. Feature dropout (prevents overfitting)
        if torch.rand(1) < 0.4 * augmentation_factor:
            mask = torch.rand_like(patches) > (self.dropout_rate * augmentation_factor)
            patches = patches * mask
        
        return patches

class ClinicalAugmentation:
    """Essential data augmentation for clinical features"""
    def __init__(self, noise_std=0.03, training=True):
        self.noise_std = noise_std
        self.training = training
    
    def __call__(self, features, is_minority_class=False):
        if not self.training:
            return features
            
        augmentation_factor = 1.3 if is_minority_class else 1.0
        
        # Adding small noise only to continuous features (age, no_instillations)
        if torch.rand(1) < 0.5 * augmentation_factor:
            noise = torch.randn_like(features) * (self.noise_std * augmentation_factor)
            # Only apply to age (position 0) and no_instillations (position 11)
            noise_mask = torch.zeros_like(features)
            noise_mask[0] = 1.0   # age
            noise_mask[11] = 1.0  # no_instillations
            features = features + noise * noise_mask
        
        return features

class BinaryBCGDataset(Dataset):
    """Binary BCG dataset: BRS3 vs BRS1/2"""
    def __init__(self, wsi_features_dir, clinical_data_dir, patient_ids=None, 
                 clinical_scaler=None, max_patches=1500, training=True, augment=True):
        self.wsi_features_dir = wsi_features_dir
        self.clinical_data_dir = clinical_data_dir
        self.max_patches = max_patches
        self.training = training
        self.augment = augment and training
        
        # Load clinical data
        self.clinical_data = self._load_clinical_data(patient_ids)
        self.clinical_df = pd.DataFrame(self.clinical_data)
        self.patient_ids = self.clinical_df['patient_id'].tolist()
        
        # Create binary labels: BRS3 vs BRS1/2
        self.labels = (self.clinical_df['BRS'] == 'BRS3').astype(int).values
        
        # Initialize augmentation
        if self.augment:
            self.wsi_augmentation = WSIPatchAugmentation(training=training)
            self.clinical_augmentation = ClinicalAugmentation(training=training)
        
        # Process clinical features
        clinical_data = []
        for _, row in self.clinical_df.iterrows():
            clinical_data.append(self._process_clinical_entry(row))
        
        clinical_data = np.array(clinical_data, dtype=np.float32)
        
        # Scale clinical features
        if clinical_scaler is None:
            self.clinical_scaler = RobustScaler()
            self.clinical_features = self.clinical_scaler.fit_transform(clinical_data)
        else:
            self.clinical_scaler = clinical_scaler
            self.clinical_features = self.clinical_scaler.transform(clinical_data)
        
        # Class distribution info
        self.class_counts = np.bincount(self.labels)
        self.minority_class = np.argmin(self.class_counts)
        #print(f"Dataset - Class 0 (BRS1/2): {self.class_counts[0]}, Class 1 (BRS3): {self.class_counts[1]}")
        #print(f"Minority class: {self.minority_class}, Augmentation: {self.augment}")

    def _load_clinical_data(self, patient_ids=None):
        """Load clinical data from JSON files"""
        clinical_data = []
        
        if patient_ids is None:
            patient_dirs = [d for d in os.listdir(self.clinical_data_dir) 
                          if os.path.isdir(os.path.join(self.clinical_data_dir, d))]
        else:
            patient_dirs = patient_ids
        
        for patient_id in patient_dirs:
            json_path = os.path.join(self.clinical_data_dir, patient_id, f"{patient_id}_CD.json")
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        patient_data = json.load(f)
                    
                    patient_data['patient_id'] = patient_id
                    
                    if 'BRS' in patient_data and patient_data['BRS'] in ['BRS1', 'BRS2', 'BRS3']:
                        clinical_data.append(patient_data)
                        
                except Exception as e:
                    print(f"Error loading {json_path}: {e}")
                    continue
        
        return clinical_data

    def _process_clinical_entry(self, row):
        """Process clinical data entry with simplified mappings"""
        
        mappings = {
            'sex': {'Male': 1, 'Female': 0, 'M': 1, 'F': 0},
            'smoking': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0},
            'tumor': {'Primary': 0, 'Recurrence': 1, 'primary': 0, 'recurrence': 1},
            'stage': {'TaHG': 0, 'T1HG': 1, 'T2HG': 2, 'Ta': 0, 'T1': 1, 'T2': 2},
            'substage': {'T1m': 0, 'T1e': 1, 'T1': 0},
            'grade': {'G2': 0, 'G3': 1, 'Low': 0, 'High': 1, 'low': 0, 'high': 1},
            'reTUR': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0},
            'LVI': {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, True: 1, False: 0},
            'variant': {'UCC': 0, 'UCC + Variant': 1, 'Pure': 0, 'Variant': 1},
            'EORTC': {'High risk': 0, 'Highest risk': 1, 'high': 0, 'highest': 1}
        }

        features = []
        feature_names = ['age', 'sex', 'smoking', 'tumor', 'stage', 'substage',
                        'grade', 'reTUR', 'LVI', 'variant', 'EORTC', 'no_instillations']
        
        for col in feature_names:
            val = row.get(col, None)
            
            if pd.isna(val) or val is None or val == '':
                # Default values for missing data, considered the median
                if col == 'age':
                    features.append(65.0)
                elif col == 'no_instillations':
                    features.append(6.0)
                else:
                    features.append(0.0)
            else:
                if col in mappings:
                    val = mappings[col].get(val, 0)
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)
                    
        return features

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]
        is_minority = (label == self.minority_class)

        # Load WSI features
        pt_file = os.path.join(self.wsi_features_dir, f"{patient_id}_HE.pt")
        try:
            features = torch.load(pt_file, map_location='cpu', weights_only=False)
            if isinstance(features, dict):
                features = list(features.values())[0]
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            
            # Ensure correct format: [num_patches, 1024]
            if features.ndim == 2 and features.shape[1] == 1024:
                patches = features
            elif features.ndim == 1 and len(features) == 1024:
                patches = features.reshape(1, -1)
            else:
                raise ValueError(f"Invalid feature format: {features.shape}")
                
        except Exception as e:
            print(f"Error loading WSI features for {patient_id}: {e}")
            patches = np.zeros((100, 1024), dtype=np.float32)

        # Convert to tensor
        patches = torch.tensor(patches, dtype=torch.float32)

        # Subsample patches if too many
        if len(patches) > self.max_patches:
            indices = torch.randperm(len(patches))[:self.max_patches]
            patches = patches[indices]
        
        # Ensure minimum number of patches
        while len(patches) < 10:
            if len(patches) > 0:
                patches = torch.cat([patches, patches[-1:]], dim=0)
            else:
                patches = torch.zeros((10, 1024), dtype=torch.float32)

        # Apply WSI augmentation
        if self.augment:
            patches = self.wsi_augmentation(patches, is_minority)

        # Get clinical features
        clinical_features = torch.tensor(self.clinical_features[idx], dtype=torch.float32)
        
        # Apply clinical augmentation
        if self.augment:
            clinical_features = self.clinical_augmentation(clinical_features, is_minority)

        return patches, clinical_features, torch.tensor(label, dtype=torch.long)