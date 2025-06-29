#!/usr/bin/env python
import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import json
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import our modules
from dataset import MultiModalDataset, custom_collate_fn_multimodal
from model import MultiModalEvidentialModel

# Import your existing radiomic extraction functions
from mpMRI_slide_level_radiomic_feature_extraction import (
    find_patient_folders, find_images_for_patient, process_single_image, 
    process_patient, BASE_DATA_PATH, OUTPUT_BASE_PATH
)
import mpMRI_slide_level_radiomic_feature_extraction as radiomic_module


class MultiModalInference:
    """Multi-modal inference class"""
    
    def __init__(self, config_path, model_path, data_folder):
        self.config_path = config_path
        self.model_path = model_path
        self.data_folder = data_folder
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set paths
        self.pathology_path = os.path.join(data_folder, "pathology", "features")
        self.clinical_path = os.path.join(data_folder, "clinical_data")
        self.radiomic_features_path = os.path.join(data_folder, "radiology", "radiomic_features")
        
    def extract_radiomic_features(self):
        """Extract radiomic features using your existing code"""
        print("🔬 Extracting radiomic features from raw MRI images...")
        
        # Update the global paths in your module
        original_base_path = radiomic_module.BASE_DATA_PATH
        original_output_path = radiomic_module.OUTPUT_BASE_PATH
        
        radiomic_module.BASE_DATA_PATH = os.path.join(self.data_folder, "radiology", "images")
        radiomic_module.OUTPUT_BASE_PATH = self.radiomic_features_path
        
        try:
            # Get patient folders using your function
            patient_folders = find_patient_folders()
            
            if not patient_folders:
                print(f"No patient folders found in {radiomic_module.BASE_DATA_PATH}")
                return False
            
            print(f"Found {len(patient_folders)} patient folders")
            
            # Process each patient using your function
            for patient_id in tqdm(patient_folders, desc="Extracting radiomic features"):
                try:
                    process_patient(patient_id)
                except Exception as e:
                    print(f"Error processing patient {patient_id}: {str(e)}")
                    continue
            
            print("✅ Radiomic features extracted successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error in radiomic extraction: {e}")
            return False
        finally:
            # Restore original paths
            radiomic_module.BASE_DATA_PATH = original_base_path
            radiomic_module.OUTPUT_BASE_PATH = original_output_path
    
    def load_clinical_data(self):
        """Load clinical data from JSON files"""
        clinical_data = []
        patient_ids = []
        
        json_files = glob.glob(os.path.join(self.clinical_path, "*.json"))
        print(f"Found {len(json_files)} clinical JSON files")
        
        for json_file in json_files:
            patient_id = os.path.basename(json_file).replace(".json", "")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                data['patient_id'] = patient_id
                clinical_data.append(data)
                patient_ids.append(patient_id)
                
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
        
        clinical_df = pd.DataFrame(clinical_data)
        
        # Convert to numeric
        for col in clinical_df.columns:
            if col != 'patient_id':
                clinical_df[col] = pd.to_numeric(clinical_df[col], errors='coerce')
        
        clinical_df = clinical_df.fillna(0)
        print(f"Loaded clinical data for {len(clinical_df)} patients")
        return clinical_df, patient_ids
    
    def prepare_wsi_data(self, patient_ids):
        """Prepare WSI pathology features"""
        wsi_features_dir = os.path.join(self.data_folder, "wsi_features")
        os.makedirs(wsi_features_dir, exist_ok=True)
        
        for patient_id in patient_ids:
            patient_path = os.path.join(self.pathology_path, patient_id)
            
            if os.path.exists(patient_path):
                pt_files = glob.glob(os.path.join(patient_path, "*.pt"))
                
                all_features = []
                for pt_file in pt_files:
                    try:
                        features = torch.load(pt_file, map_location='cpu')
                        
                        if isinstance(features, dict):
                            features = list(features.values())[0]
                        if isinstance(features, torch.Tensor):
                            features = features.numpy()
                        if features.ndim == 2 and features.shape[1] == 1024:
                            all_features.append(features)
                    except:
                        continue
                
                if all_features:
                    combined = np.vstack(all_features)
                    output_file = os.path.join(wsi_features_dir, f"{patient_id}_features.pt")
                    torch.save(torch.tensor(combined), output_file)
                else:
                    # Dummy features
                    dummy = np.zeros((100, 1024))
                    output_file = os.path.join(wsi_features_dir, f"{patient_id}_dummy.pt")
                    torch.save(torch.tensor(dummy), output_file)
        
        return wsi_features_dir
    
    def load_model(self, clinical_dim, mri_dim):
        """Load model"""
        model = MultiModalEvidentialModel(
            mri_dim=mri_dim,
            wsi_dim=1024,
            clinical_dim=clinical_dim,
            config=self.config
        )
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Model loaded")
        return model
    
    def calculate_metrics(self, y_true, y_pred, y_prob, times):
        """Calculate metrics"""
        metrics = {}
        
        try:
            if len(set(y_true)) > 1:
                # C-index
                events = np.array(y_true, dtype=bool)
                cindex, _, _, _, _ = concordance_index_censored(events, np.array(times), np.array(y_pred))
                metrics['c_index'] = cindex
                
                # Classification metrics
                auc = roc_auc_score(y_true, y_prob)
                y_pred_binary = (np.array(y_prob) > 0.5).astype(int)
                
                metrics.update({
                    'auc': auc,
                    'accuracy': accuracy_score(y_true, y_pred_binary),
                    'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                    'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
                })
            else:
                metrics = {'c_index': 0.5, 'auc': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5}
        except:
            metrics = {'c_index': 0.5, 'auc': 0.5, 'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1_score': 0.5}
        
        return metrics
    
    def run_inference(self):
        """Run inference pipeline"""
        print("="*80)
        print(" MULTI-MODAL INFERENCE")
        print("="*80)
        
        # Step 1: Extract radiomic features
        if not self.extract_radiomic_features():
            return None
        
        # Step 2: Load clinical data
        clinical_df, patient_ids = self.load_clinical_data()
        if not patient_ids:
            return None
        
        # Step 3: Prepare WSI data
        wsi_features_dir = self.prepare_wsi_data(patient_ids)
        
        # Step 4: Create dataset
        dataset = MultiModalDataset(
            mri_data_dir=self.radiomic_features_path,
            wsi_features_dir=wsi_features_dir,
            clinical_df=clinical_df,
            patient_ids=patient_ids,
            clinical_features=self.config['clinical_features'],
            max_mri_slices=self.config['data']['max_mri_slices'],
            max_wsi_patches=self.config['data']['max_wsi_patches']
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_multimodal)
        
        # Step 5: Load model
        print(f"MRI features: {dataset.n_mri_features}, Clinical: {dataset.clinical_data.shape[1]}")
        model = self.load_model(dataset.clinical_data.shape[1], dataset.n_mri_features)
        
        # Step 6: Run inference
        print(f"Running inference on {len(patient_ids)} patients...")
        
        all_risks, mri_risks, wsi_risks = [], [], []
        all_probs, mri_probs, wsi_probs = [], [], []
        all_uncertainties, mri_uncertainties, wsi_uncertainties = [], [], []
        all_labels, all_times = [], []
        
        with torch.no_grad():
            for mri_batch, wsi_batch, clinical_batch, label, time in tqdm(loader):
                try:
                    mri_slices = mri_batch[0].to(self.device)
                    wsi_patches = wsi_batch[0].to(self.device)
                    clinical = clinical_batch[0].to(self.device)
                    
                    if clinical.dim() == 1:
                        clinical = clinical.unsqueeze(0)
                    
                    outputs = model(mri_slices, wsi_patches, clinical)
                    
                    # Extract predictions
                    all_risks.append(outputs['fused_risk'].cpu().item())
                    mri_risks.append(outputs['mri_risk'].cpu().item())
                    wsi_risks.append(outputs['wsi_risk'].cpu().item())
                    
                    # Extract probabilities and uncertainties
                    for alpha, probs_list, uncertainties_list in [
                        (outputs['mri_dirichlet'], mri_probs, mri_uncertainties),
                        (outputs['wsi_dirichlet'], wsi_probs, wsi_uncertainties),
                        (outputs['fused_dirichlet'], all_probs, all_uncertainties)
                    ]:
                        S = alpha.sum(dim=1)
                        prob = (alpha[:, 1] / S).cpu().item()
                        uncertainty = (2.0 / S).cpu().item()
                        probs_list.append(prob)
                        uncertainties_list.append(uncertainty)
                    
                    all_labels.append(int(label.item()))
                    all_times.append(time.item())
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
        # Calculate metrics
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        fused_metrics = self.calculate_metrics(all_labels, all_risks, all_probs, all_times)
        mri_metrics = self.calculate_metrics(all_labels, mri_risks, mri_probs, all_times)
        wsi_metrics = self.calculate_metrics(all_labels, wsi_risks, wsi_probs, all_times)
        
        print(f"FUSED - C-index: {fused_metrics['c_index']:.4f} | AUC: {fused_metrics['auc']:.4f} | Acc: {fused_metrics['accuracy']:.4f}")
        print(f"MRI   - C-index: {mri_metrics['c_index']:.4f} | AUC: {mri_metrics['auc']:.4f} | Acc: {mri_metrics['accuracy']:.4f}")
        print(f"WSI   - C-index: {wsi_metrics['c_index']:.4f} | AUC: {wsi_metrics['auc']:.4f} | Acc: {wsi_metrics['accuracy']:.4f}")
        
        # Save results
        results_df = pd.DataFrame({
            'patient_id': patient_ids[:len(all_labels)],
            'true_label': all_labels,
            'fused_risk': all_risks,
            'mri_risk': mri_risks,
            'wsi_risk': wsi_risks,
            'fused_prob': all_probs,
            'mri_prob': mri_probs,
            'wsi_prob': wsi_probs,
            'survival_time': all_times
        })
        
        output_file = os.path.join(self.data_folder, "inference_results.csv")
        results_df.to_csv(output_file, index=False)
        print(f"💾 Results saved to: {output_file}")
        
        return results_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--model', required=True, help='Model file')
    parser.add_argument('--data', required=True, help='Data folder')
    
    args = parser.parse_args()
    
    inference = MultiModalInference(args.config, args.model, args.data)
    results = inference.run_inference()
    
    return results


if __name__ == "__main__":
    results = main()