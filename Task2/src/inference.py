import os
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

from dataset import BinaryBCGDataset, custom_collate_fn_wsi
from model import WSIEDLModel
from losses import softplus_evidence

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_all_patient_ids(clinical_data_dir):
    """Load all patient IDs from the clinical data directory"""
    patient_dirs = [d for d in os.listdir(clinical_data_dir) 
                   if os.path.isdir(os.path.join(clinical_data_dir, d))]
    
    valid_patients = []
    for patient_id in patient_dirs:
        json_path = os.path.join(clinical_data_dir, patient_id, f"{patient_id}_CD.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    patient_data = json.load(f)
                if 'BRS' in patient_data and patient_data['BRS'] in ['BRS1', 'BRS2', 'BRS3']:
                    valid_patients.append(patient_id)
            except:
                continue
    
    return valid_patients



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



def run_inference(model, loader, device):
    """Run inference on all samples"""
    model.eval()
    
    results = {
        'patient_ids': [],
        'predictions': [],
        'probabilities': [],
        'uncertainties': [],
        'true_labels': [],
        'predicted_labels': []
    }
    
    print("Running inference...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            try:
                patches_batch, clinical_batch, labels_batch = batch_data
                
                # Get patient ID (assuming single sample per batch)
                patient_id = loader.dataset.patient_ids[batch_idx]
                
                patches = patches_batch[0].to(device, non_blocking=True)
                clinical = clinical_batch[0].to(device, non_blocking=True)
                true_label = labels_batch[0].item()
                
                if clinical.dim() == 1:
                    clinical = clinical.unsqueeze(0)
                
                # Forward pass
                outputs = model(patches, clinical)
                
                # Get predictions and uncertainty from EDL
                raw_output = outputs['pred_logits']
                if raw_output.dim() > 2:
                    raw_output = raw_output.squeeze(0)
                elif raw_output.dim() == 1:
                    raw_output = raw_output.unsqueeze(0)
                
                # Convert to evidence and compute uncertainty
                evidence = softplus_evidence(raw_output)
                alpha = evidence + 1
                S = alpha.sum(dim=1)
                
                # Probability for class 1 (BRS3)
                prob_brs3 = (alpha[:, 1] / S).cpu().item()
                
                # Uncertainty calculation
                uncertainty = (2.0 / S).cpu().item()
                
                # Store results
                results['patient_ids'].append(patient_id)
                results['predictions'].append(prob_brs3)
                results['probabilities'].append(prob_brs3)
                results['uncertainties'].append(uncertainty)
                results['true_labels'].append(true_label)
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx + 1}/{len(loader)} samples")
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    return results

def calculate_metrics(results, threshold=0.5):
    """Calculate classification metrics"""
    y_true = results['true_labels']
    y_scores = results['predictions']
    
    if len(set(y_true)) < 2:
        print("Warning: Only one class present in the data")
        return {}
    
    # Calculate predicted labels using threshold
    y_pred = (np.array(y_scores) > threshold).astype(int)
    results['predicted_labels'] = y_pred.tolist()
    
    # Calculate metrics
    metrics = {}
    try:
        metrics['auc'] = roc_auc_score(y_true, y_scores)
        metrics['accuracy'] = np.mean(np.array(y_true) == y_pred)
        metrics['mean_uncertainty'] = np.mean(results['uncertainties'])
        
        # Class-wise metrics
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate precision, recall, f1 for each class
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true))
        
        metrics['precision_brs12'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        metrics['recall_brs12'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['f1_brs12'] = 2 * metrics['precision_brs12'] * metrics['recall_brs12'] / (metrics['precision_brs12'] + metrics['recall_brs12']) if (metrics['precision_brs12'] + metrics['recall_brs12']) > 0 else 0
        
        metrics['precision_brs3'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall_brs3'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_brs3'] = 2 * metrics['precision_brs3'] * metrics['recall_brs3'] / (metrics['precision_brs3'] + metrics['recall_brs3']) if (metrics['precision_brs3'] + metrics['recall_brs3']) > 0 else 0
        
        # Overall F1 (macro average)
        metrics['f1_macro'] = (metrics['f1_brs12'] + metrics['f1_brs3']) / 2
        
        metrics['confusion_matrix'] = cm
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {'error': str(e)}
    
    return metrics

def main():
    print("="*60)
    print("BCG Response Classification - Inference")
    print("="*60)
    
    # Load configuration
    try:
        config = load_config()
        print("Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device'] == 'cuda' else 'cpu')
    print(f"Device: {device}")
    
    # Check if model exists
    model_path = config['output']['model_save_path']
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load all patient IDs
    try:
        all_patient_ids = load_all_patient_ids(config['data']['clinical_data_dir'])
        print(f"Found {len(all_patient_ids)} valid patients")
    except Exception as e:
        print(f"Error loading patient IDs: {e}")
        return

    # Extract wsi features

    wsi_features_dir = prepare_wsi_data_with_slide2vec(all_patient_ids, 'uni.yaml')

    
    
    # Create dataset (using all patients)
    print("Creating dataset...")
    try:
        dataset = BinaryBCGDataset(
            wsi_features_dir=wsi_features_dir,
            clinical_data_dir=config['data']['clinical_data_dir'],
            patient_ids=all_patient_ids,
            max_patches=config['data']['max_wsi_patches'],
            training=False,
            augment=False
        )
        
        # Create data loader
        data_loader = DataLoader(
            dataset, 
            batch_size=1,  # Process one sample at a time for inference
            shuffle=False, 
            collate_fn=custom_collate_fn_wsi,
            num_workers=0
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return
    
    # Create and load model
    print("Loading model...")
    try:
        model = WSIEDLModel(
            wsi_dim=1024,
            clinical_dim=dataset.clinical_features.shape[1],
            config=config
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    try:
        results = run_inference(model, data_loader, device)
        print(f"Inference completed on {len(results['patient_ids'])} samples")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return
    
    # Calculate metrics with default threshold
    threshold = 0.5
    print(f"\nCalculating metrics with threshold = {threshold}")
    metrics = calculate_metrics(results, threshold)
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    if 'error' not in metrics:
        print(f"Total samples: {len(results['patient_ids'])}")
        print(f"Threshold: {threshold}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
        print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        
        print(f"\nClass-wise metrics:")
        print(f"BRS1/2 - Precision: {metrics['precision_brs12']:.4f}, Recall: {metrics['recall_brs12']:.4f}, F1: {metrics['f1_brs12']:.4f}")
        print(f"BRS3   - Precision: {metrics['precision_brs3']:.4f}, Recall: {metrics['recall_brs3']:.4f}, F1: {metrics['f1_brs3']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"{'':>10} {'BRS1/2':>8} {'BRS3':>8}")
        cm = metrics['confusion_matrix']
        print(f"{'BRS1/2':>10} {cm[0,0]:>8} {cm[0,1]:>8}")
        print(f"{'BRS3':>10} {cm[1,0]:>8} {cm[1,1]:>8}")
        
        # Class distribution
        true_labels = results['true_labels']
        class_counts = np.bincount(true_labels)
        print(f"\nClass distribution:")
        print(f"BRS1/2: {class_counts[0]} samples ({class_counts[0]/len(true_labels)*100:.1f}%)")
        print(f"BRS3: {class_counts[1]} samples ({class_counts[1]/len(true_labels)*100:.1f}%)")
        
    else:
        print(f"Error in metrics calculation: {metrics['error']}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'patient_id': results['patient_ids'],
        'true_label': results['true_labels'],
        'predicted_probability': results['predictions'],
        'predicted_label': results.get('predicted_labels', []),
        'uncertainty': results['uncertainties']
    })
    
    output_file = 'inference_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
