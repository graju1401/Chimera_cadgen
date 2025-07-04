import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_clinical_data_json(clinical_file_path):
    """Load clinical data from JSON file"""
    try:
        with open(clinical_file_path, 'r') as f:
            clinical_data = json.load(f)
        return clinical_data
    except Exception as e:
        print(f"Error loading clinical data from {clinical_file_path}: {e}")
        return {}

def preprocess_clinical_json(clinical_data):
    """Preprocess clinical data from JSON format for HR-NMIBC"""
    # Define categorical mappings for HR-NMIBC
    categorical_mappings = {
        'sex': {'Male': 1, 'Female': 0},
        'smoking': {'Yes': 1, 'No': 0},
        'tumor': {'Primary': 0, 'Recurrence': 1},
        'stage': {'TaHG': 0, 'T1HG': 1, 'T2HG': 2},
        'substage': {'T1m': 0, 'T1e': 1},
        'grade': {'G2': 0, 'G3': 1},
        'reTUR': {'Yes': 1, 'No': 0},
        'LVI': {'Yes': 1, 'No': 0},
        'variant': {'UCC': 0, 'UCC + Variant': 1},
        'EORTC': {'High risk': 0, 'Highest risk': 1},
        'BRS': {'BRS1': 0, 'BRS2': 1, 'BRS3': 2}
    }
    
    # Expected clinical features for HR-NMIBC
    expected_features = [
        "age", "sex", "smoking", "tumor", "stage", "substage", "grade", 
        "reTUR", "LVI", "variant", "EORTC", "no_instillations", "BRS"
    ]
    
    processed_features = []
    
    for feature in expected_features:
        if feature in clinical_data:
            value = clinical_data[feature]
            
            # Apply categorical mapping if available
            if feature in categorical_mappings and value in categorical_mappings[feature]:
                processed_value = categorical_mappings[feature][value]
            else:
                # Try to convert to numeric
                try:
                    processed_value = float(value)
                    # Handle special case for no_instillations (-1 means missing)
                    if feature == 'no_instillations' and processed_value == -1:
                        processed_value = 0.0  # or use median/mean
                except:
                    processed_value = 0.0  # Default value for missing
            
            processed_features.append(processed_value)
        else:
            processed_features.append(0.0)  # Default for missing features
    
    return np.array(processed_features, dtype=np.float32)

def load_rna_data(rna_file_path):
    """Load and preprocess RNA-seq data from JSON"""
    try:
        with open(rna_file_path, 'r') as f:
            rna_data = json.load(f)
        
        # Convert to numpy array
        gene_names = list(rna_data.keys())
        gene_values = np.array(list(rna_data.values()), dtype=np.float32)
        
        # Log2 transform (add 1 to avoid log(0))
        gene_values = np.log2(gene_values + 1.0)
        
        return gene_values.reshape(-1, 1), gene_names  # (num_genes, 1)
    except Exception as e:
        print(f"Error loading RNA data from {rna_file_path}: {e}")
        return np.zeros((1000, 1), dtype=np.float32), []

class MultiModalDataset(Dataset):
    def __init__(self, data_dir, patient_ids, clinical_scaler=None, rna_scaler=None, max_wsi_patches=20000):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.max_wsi_patches = max_wsi_patches
        
        # Load all clinical data and target variables
        all_clinical_features = []
        self.progression = []
        self.times = []
        
        for patient_id in patient_ids:
            # Load clinical data from JSON
            clinical_file = os.path.join(data_dir, str(patient_id), f"{patient_id}_CD.json")
            if os.path.exists(clinical_file):
                clinical_data = load_clinical_data_json(clinical_file)
                processed_features = preprocess_clinical_json(clinical_data)
                all_clinical_features.append(processed_features)
                
                # Extract target variables for HR-NMIBC
                self.progression.append(float(clinical_data.get('progression', 0)))
                self.times.append(float(clinical_data.get('time_to_HG_recur_or_FUend', 12.0)))
            else:
                # Default values if clinical file not found
                all_clinical_features.append(np.zeros(13, dtype=np.float32))
                self.progression.append(0.0)
                self.times.append(12.0)
                print(f"Clinical data not found for patient {patient_id}")
        
        self.progression = np.array(self.progression)
        self.times = np.array(self.times)
        
        # Scale clinical features
        clinical_data_matrix = np.array(all_clinical_features)
        if clinical_scaler is None:
            self.clinical_scaler = RobustScaler()
            self.clinical_features = self.clinical_scaler.fit_transform(clinical_data_matrix)
        else:
            self.clinical_scaler = clinical_scaler
            self.clinical_features = self.clinical_scaler.transform(clinical_data_matrix)
        
        # RNA scaler
        if rna_scaler is None:
            self.rna_scaler = RobustScaler()
            self._fit_rna_scaler()
        else:
            self.rna_scaler = rna_scaler
        
    
    def _fit_rna_scaler(self):
        """Fit RNA scaler on all available RNA data"""
        all_gene_values = []
        for pid in self.patient_ids:
            rna_file = os.path.join(self.data_dir, str(pid), f"{pid}_RNA.json")
            if os.path.exists(rna_file):
                gene_values, _ = load_rna_data(rna_file)
                all_gene_values.append(gene_values)
        
        if all_gene_values:
            all_data = np.vstack(all_gene_values)
            self.rna_scaler.fit(all_data)
            print(f"RNA scaler fitted on {len(all_gene_values)} samples")
        else:
            print("No RNA data found for scaler fitting")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load RNA data
        rna_file = os.path.join(self.data_dir, str(patient_id), f"{patient_id}_RNA.json")
        if os.path.exists(rna_file):
            gene_values, gene_names = load_rna_data(rna_file)
            gene_values = self.rna_scaler.transform(gene_values)
        else:
            gene_values = np.zeros((1000, 1), dtype=np.float32)
            gene_names = [f"gene_{i}" for i in range(1000)]
        
        # Load WSI features
        wsi_features_file = os.path.join(self.data_dir, "features", "features", f"{patient_id}_HE.pt")
        if os.path.exists(wsi_features_file):
            try:
                wsi_features = torch.load(wsi_features_file, map_location='cpu')
                if isinstance(wsi_features, dict):
                    wsi_features = list(wsi_features.values())[0]
                if isinstance(wsi_features, torch.Tensor):
                    wsi_features = wsi_features.numpy()
                
                # Subsample patches if too many
                if len(wsi_features) > self.max_wsi_patches:
                    indices = np.random.choice(len(wsi_features), self.max_wsi_patches, replace=False)
                    wsi_features = wsi_features[indices]
            except Exception as e:
                print(f"Error loading WSI features for {patient_id}: {e}")
                wsi_features = np.zeros((100, 1024), dtype=np.float32)
        else:
            wsi_features = np.zeros((100, 1024), dtype=np.float32)
        
        return (
            torch.tensor(gene_values, dtype=torch.float32),
            torch.tensor(wsi_features, dtype=torch.float32),
            torch.tensor(self.clinical_features[idx], dtype=torch.float32),
            torch.tensor(self.progression[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32)
        )

def multimodal_collate_fn(batch):
    """Custom collate function for variable length data"""
    rna_list = []
    wsi_list = []
    clinical_list = []
    labels_list = []
    times_list = []
    
    for rna, wsi, clinical, label, time in batch:
        rna_list.append(rna)
        wsi_list.append(wsi)
        clinical_list.append(clinical)
        labels_list.append(label)
        times_list.append(time)
    
    # Stack fixed-size tensors
    clinical_batch = torch.stack(clinical_list)
    labels_batch = torch.stack(labels_list)
    times_batch = torch.stack(times_list)
    
    # Keep variable-length data as lists
    return rna_list, wsi_list, clinical_batch, labels_batch, times_batch