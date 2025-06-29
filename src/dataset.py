import torch
from torch.utils.data import Dataset
import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
from typing import List, Optional, Tuple, Union

class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    
    def __init__(self, clinical_df: pd.DataFrame, patient_ids: List[int], 
                 clinical_features: List[str], clinical_scaler: Optional[RobustScaler] = None):
        self.patient_ids = patient_ids
        self.clinical_features = clinical_features
        
        # Process clinical data
        patient_clinical = clinical_df[clinical_df['patient_id'].isin(patient_ids)].copy()
        patient_clinical = patient_clinical.set_index('patient_id').loc[patient_ids].reset_index()
        
        self.labels = patient_clinical['BCR'].values.astype(float)
        
        # Handle time data
        time_col = next((col for col in ['Time to last follow-up / BCR', 'time_to_BCR', 'survival_time'] 
                        if col in clinical_df.columns), None)
        if time_col:
            self.times = patient_clinical[time_col].values.astype(float)
        else:
            np.random.seed(42)
            self.times = np.where(self.labels == 1, np.random.exponential(24, len(self.labels)),
                                 np.random.exponential(60, len(self.labels)))
        
        # Process clinical features
        available_cols = [col for col in clinical_features if col in clinical_df.columns]
        patient_clinical_clean = self._preprocess_clinical_data(patient_clinical, available_cols)
        clinical_data = patient_clinical_clean[available_cols].values.astype(np.float32)
        
        if clinical_scaler is None:
            self.clinical_scaler = RobustScaler()
            self.clinical_data = self.clinical_scaler.fit_transform(clinical_data)
        else:
            self.clinical_scaler = clinical_scaler
            self.clinical_data = self.clinical_scaler.transform(clinical_data)
    
    def _preprocess_clinical_data(self, df: pd.DataFrame, clinical_cols: List[str]) -> pd.DataFrame:
        """Preprocess clinical data to handle non-numeric values"""
        processed_df = df.copy()
        
        for col in clinical_cols:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str)
                
                processed_df[col] = processed_df[col].replace({
                    'x': np.nan, 'X': np.nan, 'nan': np.nan, 'NaN': np.nan,
                    'NA': np.nan, 'N/A': np.nan, '': np.nan, ' ': np.nan,
                    'unknown': np.nan, 'Unknown': np.nan, 'UNKNOWN': np.nan,
                    'missing': np.nan, 'Missing': np.nan, 'MISSING': np.nan,
                    'none': np.nan, 'None': np.nan, 'NONE': np.nan,
                    '.': np.nan, '?': np.nan, '-': np.nan, '--': np.nan
                })
                
                try:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    non_null_mask = processed_df[col].notna()
                    if non_null_mask.sum() > 0:
                        processed_df.loc[non_null_mask, col] = le.fit_transform(processed_df.loc[non_null_mask, col])
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                
                processed_df[col] = processed_df[col].fillna(0)
        
        return processed_df
    
    def __len__(self):
        return len(self.patient_ids)


class MRIDataset(BaseDataset):
    """Dataset for MRI + Clinical data"""
    
    def __init__(self, mri_data_dir: str, clinical_df: pd.DataFrame, patient_ids: List[int],
                 clinical_features: List[str], clinical_scaler: Optional[RobustScaler] = None,
                 mri_scaler: Optional[RobustScaler] = None, max_slices: int = 50):
        super().__init__(clinical_df, patient_ids, clinical_features, clinical_scaler)
        
        self.mri_data_dir = mri_data_dir
        self.max_slices = max_slices
        
        # Detect MRI features
        self._detect_mri_features()
        
        # MRI scaler
        if mri_scaler is None:
            self.mri_scaler = RobustScaler()
            self._fit_mri_scaler()
        else:
            self.mri_scaler = mri_scaler
        
        print(f"MRI Dataset initialized: {len(self.patient_ids)} patients")
        print(f"MRI features: {self.n_mri_features}")
        print(f"Clinical features: {len(clinical_features)}")
    
    def _detect_mri_features(self):
        for pid in self.patient_ids:
            folder = os.path.join(self.mri_data_dir, str(pid))
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.csv"))
                if files:
                    try:
                        df = pd.read_csv(files[0])
                        for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                            if col in df.columns:
                                df = df.drop(columns=[col])
                        self.n_mri_features = df.shape[1]
                        return
                    except:
                        continue
        self.n_mri_features = 100
    
    def _fit_mri_scaler(self):
        all_features = []
        for pid in self.patient_ids:
            folder = os.path.join(self.mri_data_dir, str(pid))
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.csv"))
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                            if col in df.columns:
                                df = df.drop(columns=[col])
                        if len(df.columns) > 0:
                            all_features.append(df.values)
                    except:
                        continue
        
        if all_features:
            all_data = np.vstack(all_features)
            self.mri_scaler.fit(all_data)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load MRI data
        folder = os.path.join(self.mri_data_dir, str(patient_id))
        all_slices = []
        
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, "*.csv"))
            for file in files:
                try:
                    df = pd.read_csv(file)
                    for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                        if col in df.columns:
                            df = df.drop(columns=[col])
                    if len(df.columns) > 0:
                        all_slices.append(df.values)
                except:
                    continue
        
        if all_slices:
            slices = np.vstack(all_slices)
            if len(slices) > self.max_slices:
                indices = np.random.choice(len(slices), self.max_slices, replace=False)
                slices = slices[indices]
            slices = self.mri_scaler.transform(slices)
        else:
            slices = np.zeros((10, self.n_mri_features))
        
        return (
            torch.tensor(slices, dtype=torch.float32),
            torch.tensor(self.clinical_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32)
        )


class WSIDataset(BaseDataset):
    """Dataset for WSI + Clinical data"""
    
    def __init__(self, wsi_features_dir: str, clinical_df: pd.DataFrame, patient_ids: List[int],
                 clinical_features: List[str], clinical_scaler: Optional[RobustScaler] = None,
                 max_patches: int = 20000):
        super().__init__(clinical_df, patient_ids, clinical_features, clinical_scaler)
        
        self.wsi_features_dir = wsi_features_dir
        self.max_patches = max_patches
        
        print(f"WSI Dataset initialized: {len(self.patient_ids)} patients")
        print(f"Clinical features: {len(clinical_features)}")
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load WSI data
        wsi_files = glob.glob(os.path.join(self.wsi_features_dir, f"{patient_id}_*.pt"))
        all_patches = []
        
        for file in wsi_files:
            try:
                features = torch.load(file, map_location='cpu')
                if isinstance(features, dict):
                    features = list(features.values())[0]
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                if features.ndim == 2 and features.shape[1] == 1024:
                    all_patches.append(features)
            except:
                continue
        
        if all_patches:
            patches = np.vstack(all_patches)
            if len(patches) > self.max_patches:
                indices = np.random.choice(len(patches), self.max_patches, replace=False)
                patches = patches[indices]
        else:
            patches = np.zeros((100, 1024))
        
        return (
            torch.tensor(patches, dtype=torch.float32),
            torch.tensor(self.clinical_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32)
        )


class MultiModalDataset(BaseDataset):
    """Dataset for MRI + WSI + Clinical data"""
    
    def __init__(self, mri_data_dir: str, wsi_features_dir: str, clinical_df: pd.DataFrame, 
                 patient_ids: List[int], clinical_features: List[str],
                 clinical_scaler: Optional[RobustScaler] = None, mri_scaler: Optional[RobustScaler] = None,
                 max_mri_slices: int = 50, max_wsi_patches: int = 20000):
        super().__init__(clinical_df, patient_ids, clinical_features, clinical_scaler)
        
        self.mri_data_dir = mri_data_dir
        self.wsi_features_dir = wsi_features_dir
        self.max_mri_slices = max_mri_slices
        self.max_wsi_patches = max_wsi_patches
        
        # Detect MRI features
        self._detect_mri_features()
        
        # MRI scaler
        if mri_scaler is None:
            self.mri_scaler = RobustScaler()
            self._fit_mri_scaler()
        else:
            self.mri_scaler = mri_scaler
        
        print(f"MultiModal Dataset initialized: {len(self.patient_ids)} patients")
        print(f"MRI features: {self.n_mri_features}")
        print(f"Clinical features: {len(clinical_features)}")
    
    def _detect_mri_features(self):
        for pid in self.patient_ids:
            folder = os.path.join(self.mri_data_dir, str(pid))
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.csv"))
                if files:
                    try:
                        df = pd.read_csv(files[0])
                        for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                            if col in df.columns:
                                df = df.drop(columns=[col])
                        self.n_mri_features = df.shape[1]
                        return
                    except:
                        continue
        self.n_mri_features = 100
    
    def _fit_mri_scaler(self):
        all_features = []
        for pid in self.patient_ids:
            folder = os.path.join(self.mri_data_dir, str(pid))
            if os.path.exists(folder):
                files = glob.glob(os.path.join(folder, "*.csv"))
                for file in files:
                    try:
                        df = pd.read_csv(file)
                        for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                            if col in df.columns:
                                df = df.drop(columns=[col])
                        if len(df.columns) > 0:
                            all_features.append(df.values)
                    except:
                        continue
        
        if all_features:
            all_data = np.vstack(all_features)
            self.mri_scaler.fit(all_data)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # Load MRI data
        mri_folder = os.path.join(self.mri_data_dir, str(patient_id))
        all_mri_slices = []
        if os.path.exists(mri_folder):
            files = glob.glob(os.path.join(mri_folder, "*.csv"))
            for file in files:
                try:
                    df = pd.read_csv(file)
                    for col in ['patient_id', 'slice_id', 'sequence', 'modality']:
                        if col in df.columns:
                            df = df.drop(columns=[col])
                    if len(df.columns) > 0:
                        all_mri_slices.append(df.values)
                except:
                    continue
        
        if all_mri_slices:
            mri_slices = np.vstack(all_mri_slices)
            if len(mri_slices) > self.max_mri_slices:
                indices = np.random.choice(len(mri_slices), self.max_mri_slices, replace=False)
                mri_slices = mri_slices[indices]
            mri_slices = self.mri_scaler.transform(mri_slices)
        else:
            mri_slices = np.zeros((10, self.n_mri_features))
        
        # Load WSI data
        wsi_files = glob.glob(os.path.join(self.wsi_features_dir, f"{patient_id}_*.pt"))
        all_wsi_patches = []
        
        for file in wsi_files:
            try:
                features = torch.load(file, map_location='cpu')
                if isinstance(features, dict):
                    features = list(features.values())[0]
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                if features.ndim == 2 and features.shape[1] == 1024:
                    all_wsi_patches.append(features)
            except:
                continue
        
        if all_wsi_patches:
            wsi_patches = np.vstack(all_wsi_patches)
            if len(wsi_patches) > self.max_wsi_patches:
                indices = np.random.choice(len(wsi_patches), self.max_wsi_patches, replace=False)
                wsi_patches = wsi_patches[indices]
        else:
            wsi_patches = np.zeros((100, 1024))
        
        return (
            torch.tensor(mri_slices, dtype=torch.float32),
            torch.tensor(wsi_patches, dtype=torch.float32),
            torch.tensor(self.clinical_data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.times[idx], dtype=torch.float32)
        )


def custom_collate_fn_mri(batch):
    """Custom collate function for MRI variable length data"""
    slices_list = []
    clinical_list = []
    labels_list = []
    times_list = []
    
    for slices, clinical, label, time in batch:
        slices_list.append(slices)
        clinical_list.append(clinical)
        labels_list.append(label)
        times_list.append(time)
    
    clinical_batch = torch.stack(clinical_list)
    labels_batch = torch.stack(labels_list)
    times_batch = torch.stack(times_list)
    
    return slices_list, clinical_batch, labels_batch, times_batch


def custom_collate_fn_wsi(batch):
    """Custom collate function for WSI variable length data"""
    patches_list = []
    clinical_list = []
    labels_list = []
    times_list = []
    
    for patches, clinical, label, time in batch:
        patches_list.append(patches)
        clinical_list.append(clinical)
        labels_list.append(label)
        times_list.append(time)
    
    clinical_batch = torch.stack(clinical_list)
    labels_batch = torch.stack(labels_list)
    times_batch = torch.stack(times_list)
    
    return patches_list, clinical_batch, labels_batch, times_batch


def custom_collate_fn_multimodal(batch):
    """Custom collate function for multi-modal variable length data"""
    mri_list = []
    wsi_list = []
    clinical_list = []
    labels_list = []
    times_list = []
    
    for mri, wsi, clinical, label, time in batch:
        mri_list.append(mri)
        wsi_list.append(wsi)
        clinical_list.append(clinical)
        labels_list.append(label)
        times_list.append(time)
    
    clinical_batch = torch.stack(clinical_list)
    labels_batch = torch.stack(labels_list)
    times_batch = torch.stack(times_list)
    
    return mri_list, wsi_list, clinical_batch, labels_batch, times_batch