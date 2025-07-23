import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, 
                           classification_report)
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

# Import from your local modules
from dataset_infe import MultiModalDataset, multimodal_collate_fn, load_clinical_data_json
from model import MultiModalEvidentialModel

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_all_patient_ids(data_dir):
    """Get all available patient IDs"""
    patient_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and d != 'features']
    patient_ids = []
    
    for patient_id in patient_dirs:
        clinical_file = os.path.join(data_dir, patient_id, f"{patient_id}_CD.json")
        if os.path.exists(clinical_file):
            patient_ids.append(patient_id)
    
    return patient_ids

def inference_evaluation(model, loader, device):
    """Comprehensive inference evaluation"""
    model.eval()
    
    # Storage for results
    results = {
        'rna_risks': [], 'wsi_risks': [], 'fused_risks': [],
        'rna_probs': [], 'wsi_probs': [], 'fused_probs': [],
        'rna_uncertainties': [], 'wsi_uncertainties': [], 'fused_uncertainties': [],
        'labels': [], 'times': [], 'patient_ids': []
    }
    
    print("Running inference on all patients...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            try:
                if len(batch_data) == 5:
                    rna_batch, wsi_batch, clinical_batch, labels_batch, times_batch = batch_data
                    
                    # Process single patient (batch_size=1 for inference)
                    gene_expressions = rna_batch[0].to(device)
                    wsi_patches = wsi_batch[0].to(device)
                    clinical = clinical_batch[0].to(device)
                    label = labels_batch[0].item()
                    time = times_batch[0].item()
                    
                    if clinical.dim() == 1:
                        clinical = clinical.unsqueeze(0)
                    
                    # Model inference
                    outputs = model(gene_expressions, wsi_patches, clinical)
                    
                    # Extract risks
                    results['rna_risks'].append(outputs['rna_risk'].cpu().item())
                    results['wsi_risks'].append(outputs['wsi_risk'].cpu().item())
                    results['fused_risks'].append(outputs['fused_risk'].cpu().item())
                    
                    # Extract probabilities and uncertainties from Dirichlet distributions
                    for modality, alpha_key in [('rna', 'rna_dirichlet'), 
                                              ('wsi', 'wsi_dirichlet'), 
                                              ('fused', 'fused_dirichlet')]:
                        alpha = outputs[alpha_key]
                        if alpha.dim() > 2:
                            alpha = alpha.squeeze(0)
                        
                        # Calculate probability of progression (class 1)
                        S = alpha.sum(dim=1)
                        prob = (alpha[:, 1] / S).cpu().item()
                        uncertainty = (2.0 / S).cpu().item()  # Epistemic uncertainty
                        
                        results[f'{modality}_probs'].append(prob)
                        results[f'{modality}_uncertainties'].append(uncertainty)
                    
                    results['labels'].append(int(label))
                    results['times'].append(time)
                    results['patient_ids'].append(batch_idx)  # You can modify this to get actual patient IDs
                    
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Processed {batch_idx + 1} patients...")
                        
            except Exception as e:
                print(f"Error processing patient {batch_idx}: {e}")
                continue
    
    return results

def calculate_comprehensive_metrics(results):
    """Calculate all classification and survival metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*80)
    
    labels = np.array(results['labels'])
    times = np.array(results['times'])
    
    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"Warning: Only one class present in labels: {unique_labels}")
        return {}
    
    all_metrics = {}
    
    # Evaluate each modality
    modalities = ['rna', 'wsi', 'fused']
    
    for modality in modalities:
        print(f"\n {modality.upper()} MODALITY RESULTS:")
        print("-" * 50)
        
        risks = np.array(results[f'{modality}_risks'])
        probs = np.array(results[f'{modality}_probs'])
        uncertainties = np.array(results[f'{modality}_uncertainties'])
        
        # Binary predictions
        pred_labels = (probs > 0.5).astype(int)
        
        try:
            # Survival Analysis - Concordance Index (Primary Metric)
            cindex, _, _, _, _ = concordance_index_censored(
                labels.astype(bool), times, risks
            )
            print(f"C-Index (Survival): {cindex:.4f}")
            
            # Classification Metrics
            auc = roc_auc_score(labels, probs)
            accuracy = accuracy_score(labels, pred_labels)
            precision = precision_score(labels, pred_labels, zero_division=0)
            recall = recall_score(labels, pred_labels, zero_division=0)
            f1 = f1_score(labels, pred_labels, zero_division=0)
            
            print(f"AUC-ROC: {auc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall (Sensitivity): {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Specificity
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            print(f"Specificity: {specificity:.4f}")
            
            # Evidential Uncertainty
            mean_uncertainty = np.mean(uncertainties)
            print(f"Mean Uncertainty: {mean_uncertainty:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(labels, pred_labels)
            print(f"Confusion Matrix:")
            print(f"  TN: {tn}, FP: {fp}")
            print(f"  FN: {fn}, TP: {tp}")
            
            # Store metrics
            all_metrics[modality] = {
                'cindex': cindex,
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'mean_uncertainty': mean_uncertainty,
                'confusion_matrix': cm
            }
            
        except Exception as e:
            print(f"Error calculating metrics for {modality}: {e}")
            all_metrics[modality] = {}
    
    # Compare modalities
    print(f"\n MULTIMODAL FUSION ANALYSIS:")
    print("-" * 50)
    
    if 'fused' in all_metrics and 'rna' in all_metrics and 'wsi' in all_metrics:
        fused_cindex = all_metrics['fused'].get('cindex', 0)
        rna_cindex = all_metrics['rna'].get('cindex', 0)
        wsi_cindex = all_metrics['wsi'].get('cindex', 0)
        
        print(f"C-Index Improvement over RNA: {fused_cindex - rna_cindex:+.4f}")
        print(f"C-Index Improvement over WSI: {fused_cindex - wsi_cindex:+.4f}")
        
        fused_auc = all_metrics['fused'].get('auc', 0)
        rna_auc = all_metrics['rna'].get('auc', 0)
        wsi_auc = all_metrics['wsi'].get('auc', 0)
        
        print(f"AUC Improvement over RNA: {fused_auc - rna_auc:+.4f}")
        print(f"AUC Improvement over WSI: {fused_auc - wsi_auc:+.4f}")
    
    # Sample-level analysis
    print(f"\n DATASET SUMMARY:")
    print("-" * 50)
    print(f"Total Patients: {len(labels)}")
    print(f"Progression Events: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    print(f"No Progression: {len(labels) - np.sum(labels)} ({(1-np.mean(labels))*100:.1f}%)")
    print(f"Mean Time to Event/Censoring: {np.mean(times):.2f}")
    
    return all_metrics


def prepare_wsi_data_with_slide2vec(patient_ids, config_file: str):
    """
    Prepares WSI data using Slide2Vec, outputting tile-level features.

    This method automatically generates a slide path CSV and updates the
    config file before running the feature extraction pipeline. It is
    designed for tile-level models (e.g., UNI) and returns the path
    to the directory containing per-slide tile feature files.

    Args:
        patient_ids (list): List of patient IDs to process.
        config_file (str): Path to the base Slide2Vec config YAML file.

    Returns:
        str: Path to the directory containing tile-level features.
                This directory will contain files like `[slide_name].pt`,
                where each file holds a tensor of features for all tiles
                from that slide.
    """

    # 1. Find all WSI paths and create a CSV with an empty mask column
    data_folder = "/data"
    csv_path = Path(data_folder) / "slide_paths.csv"
    wsi_files = []

    print("Searching for WSI files (ignoring masks)...")
    for patient_id in patient_ids:
        patient_path = Path(data_folder) / patient_id
        for root, _, files in os.walk(patient_path):
            for file in files:
                file_lower = file.lower()
                
                # Condition: The file must be a WSI format AND must NOT be a mask file.
                is_wsi = file_lower.endswith((".svs", ".ndpi", ".tif"))
                is_mask = file_lower.endswith("_mask.tif")
                
                if is_wsi and not is_mask:
                    wsi_path = Path(root) / file
                    # Add the WSI path and an empty string for the mask path
                    wsi_files.append({
                        "wsi_path": str(wsi_path),
                        "mask_path": ""  # Always provide an empty mask path
                    })
        if len(wsi_files) > 3:
            break
    if not wsi_files:
        raise FileNotFoundError("No WSI files (.svs, .tif, .ndpi) found for the given patient IDs.")

    # Create a DataFrame and save it to CSV with the correct columns
    df = pd.DataFrame(wsi_files)
    # Ensure the columns are in the correct order for the slide2vec pipeline
    df.to_csv(csv_path, index=False, columns=['wsi_path'])

    print(f"Successfully generated CSV with {len(df)} WSI files (no masks) at: {csv_path}")
        
    # 2. Load and modify the config YAML
    with open(config_file, 'r') as f:
        cfg_data = yaml.safe_load(f)

    cfg_data["csv"] = str(csv_path)

    output_dir = Path(data_folder) / "slide2vec_output"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
    cfg_data["output_dir"] = str(output_dir)

    # 3. Save modified config to a temp file
    updated_config_path = Path(data_folder) / "temp_slide2vec_config.yaml"
    with open(updated_config_path, 'w') as f:
        yaml.safe_dump(cfg_data, f)
        
    # Set the token for the subprocess environment
    # First, get the token from the environment of THIS script
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # This check is good, let's keep it.
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not found.")

    # Create a copy of the current environment for the subprocess
    sub_env = os.environ.copy()
    
    # ** THE FIX: Set BOTH possible environment variable names **
    # The modern name used by huggingface_hub
    sub_env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    # The older/shorter name that some libraries still look for
    sub_env["HF_TOKEN"] = hf_token

    # 4. Call Slide2Vec pipeline, PASSING THE MODIFIED ENVIRONMENT
    print(f"Running Slide2Vec feature extraction. Output will be in {output_dir}")
    
    result = subprocess.run([
        "python", "-m", "slide2vec.main",
        "--config-file", str(updated_config_path)
    ], capture_output=True, text=True, env=sub_env) # <-- The env argument is key

    if result.returncode != 0:
        print("--- Slide2Vec STDOUT ---")
        print(result.stdout)
        print("--- Slide2Vec STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Slide2Vec pipeline failed with exit code {result.returncode}.")

    # 5. Return the path to the TILE-LEVEL features

    # Find the latest run directory created by slide2vec inside the output_dir.
    # This handles the timestamped folder (e.g., '2025-07-19_11_21').
    try:
        run_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            # This error will be caught by the except block below
            raise FileNotFoundError("No run directories found in the output directory.")

        # Find the most recently modified directory, which corresponds to the latest run
        latest_run_dir = max(run_dirs, key=os.path.getmtime)
        
        # Construct the correct path to the features
        tile_features_dir = latest_run_dir / "features"
        print(f"Searching for features in latest run directory: {tile_features_dir}")

    except (FileNotFoundError, ValueError) as e:
        # This block will catch errors if output_dir is empty or doesn't exist.
        print("--- Slide2Vec STDOUT ---")
        print(result.stdout)
        print("--- Slide2Vec STDERR ---")
        print(result.stderr)
        raise FileNotFoundError(
            f"Could not locate a run directory inside '{output_dir}'. "
            f"The slide2vec pipeline may have failed to produce output. Original error: {e}"
        )

    # Now, check if the final, correctly identified path exists
    if not tile_features_dir.exists():
        print("--- Slide2Vec STDOUT ---")
        print(result.stdout)
        print("--- Slide2Vec STDERR ---")
        print(result.stderr)
        raise FileNotFoundError(
            f"Tile-level features directory not found at the expected path: '{tile_features_dir}'. "
            "Check the Slide2Vec output and config. The pipeline might have failed "
            "or the output structure might have changed."
        )

    print(f"Successfully generated tile-level features in: {tile_features_dir}")
    return str(tile_features_dir)


def save_detailed_results(results, all_metrics, output_dir="inference_results"):
    """Save detailed results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save patient-level predictions
    df_predictions = pd.DataFrame({
        'patient_id': results['patient_ids'],
        'true_label': results['labels'],
        'time': results['times'],
        'rna_risk': results['rna_risks'],
        'rna_prob': results['rna_probs'],
        'rna_uncertainty': results['rna_uncertainties'],
        'wsi_risk': results['wsi_risks'],
        'wsi_prob': results['wsi_probs'],
        'wsi_uncertainty': results['wsi_uncertainties'],
        'fused_risk': results['fused_risks'],
        'fused_prob': results['fused_probs'],
        'fused_uncertainty': results['fused_uncertainties']
    })
    
    predictions_file = os.path.join(output_dir, "patient_predictions.csv")
    df_predictions.to_csv(predictions_file, index=False)
    print(f"\nPatient-level predictions saved to: {predictions_file}")
    
    # Save metrics summary
    metrics_summary = []
    for modality, metrics in all_metrics.items():
        if metrics:  # Check if metrics dict is not empty
            metrics_summary.append({
                'modality': modality,
                **metrics
            })
    
    if metrics_summary:
        df_metrics = pd.DataFrame(metrics_summary)
        metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
        df_metrics.to_csv(metrics_file, index=False)
        print(f"Evaluation metrics saved to: {metrics_file}")

def main():
    print(" HR-NMIBC Multimodal Evidential Learning - INFERENCE")
    print("="*80)
    
    # Configuration
    config_path = 'config.yaml'  # Adjust path as needed
    model_path = 'best_hrnmibc_evidential_model.pth'  # Adjust path as needed
    
    # Load configuration
    config = load_config(config_path)
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get all patient IDs
    patient_ids = get_all_patient_ids(config['data']['data_dir'])
    print(f"Found {len(patient_ids)} patients for inference")
    
    if len(patient_ids) == 0:
        print(" No patients found! Check your data directory.")
        return

    # Extract wsi features
    wsi_features_dir = prepare_wsi_data_with_slide2vec(patient_ids, 'uni.yaml')
    
    # Create dataset (we'll use all patients for inference)
    print("Loading dataset...")
    dataset = MultiModalDataset(
        config['data']['data_dir'], 
        patient_ids,
        wsi_features_dir,
        max_wsi_patches=config['data']['max_wsi_patches']
    )
    
    # Create data loader (batch_size=1 for inference)
    data_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=multimodal_collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultiModalEvidentialModel(
        wsi_dim=config['model']['wsi_encoder']['input_dim'],
        clinical_dim=dataset.clinical_features.shape[1],
        config=config
    ).to(device)
    
    # Load trained model
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Model loaded successfully!")
    else:
        print(f" Model file not found: {model_path}")
        print("Please ensure the trained model file exists.")
        return
    
    # Run inference
    print(f"\n Starting inference on {len(patient_ids)} patients...")
    results = inference_evaluation(model, data_loader, device)
    
    if len(results['labels']) == 0:
        print(" No valid predictions generated!")
        return
    
    print(f" Successfully processed {len(results['labels'])} patients")
    
    # Calculate comprehensive metrics
    all_metrics = calculate_comprehensive_metrics(results)
    
    # Save detailed results
    save_detailed_results(results, all_metrics)
    
    print("\n Inference completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
