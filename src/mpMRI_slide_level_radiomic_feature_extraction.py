import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from radiomics import featureextractor
from PIL import Image
import io
import glob
from tqdm import tqdm
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress PyRadiomics logging warnings
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.glcm').setLevel(logging.ERROR)
logging.getLogger('radiomics.shape').setLevel(logging.ERROR)
logging.getLogger('radiomics.shape2D').setLevel(logging.ERROR)

# -----------------------------
# Configuration
# -----------------------------
BASE_DATA_PATH = "data/radiology/images"
OUTPUT_BASE_PATH = "data/radiology/radiomic_features"
MODALITIES = ["t2w", "adc", "hbv"]  # Multi-parametric MRI modalities

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

# -----------------------------
# Utility Functions
# -----------------------------
def find_slices_with_masks(mask_np):
    """Find all slices that contain mask data."""
    slices_with_masks = []
    
    for slice_idx in range(mask_np.shape[0]):
        mask_slice = mask_np[slice_idx]
        if np.sum(mask_slice) > 0:  # If slice has any mask pixels
            slices_with_masks.append(slice_idx)
    
    return slices_with_masks


def resample_mask(mask, reference_image):
    """Resample mask to match reference image geometry."""
    try:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputSpacing(reference_image.GetSpacing())
        resampler.SetOutputOrigin(reference_image.GetOrigin())
        resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetSize(reference_image.GetSize())
        return resampler.Execute(mask)
    except Exception as e:
        print(f"[!] Error resampling mask: {str(e)}")
        return mask


def extract_radiomic_features_2d(image_slice, mask_slice):
    """Extract radiomic features from a 2D slice."""
    try:
        # Convert 2D arrays to SimpleITK images
        image_sitk = sitk.GetImageFromArray(image_slice)
        mask_sitk = sitk.GetImageFromArray(mask_slice.astype(int))
        
        # Set same spacing and origin for both
        image_sitk.SetSpacing([1.0, 1.0])
        mask_sitk.SetSpacing([1.0, 1.0])
        image_sitk.SetOrigin([0.0, 0.0])
        mask_sitk.SetOrigin([0.0, 0.0])
        
        # Suppress PyRadiomics warnings during extraction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            extractor = featureextractor.RadiomicsFeatureExtractor()
            # Only extract Original image type features
            extractor.enableImageTypeByName('Original')
            extractor.enableAllFeatures()
            
            # Set 2D mode
            extractor.settings['force2D'] = True
            extractor.settings['force2Ddimension'] = 0
            
            # Additional settings to reduce warnings
            extractor.settings['verbose'] = False
            
            result = extractor.execute(image_sitk, mask_sitk)
        
        return {k: v for k, v in result.items() if not k.startswith("diagnostics")}
    
    except Exception as e:
        print(f"[!] Error extracting radiomic features: {str(e)}")
        return {}


def find_patient_folders():
    """Find all patient folders in the base data path."""
    patient_folders = []
    if os.path.exists(BASE_DATA_PATH):
        for item in os.listdir(BASE_DATA_PATH):
            patient_path = os.path.join(BASE_DATA_PATH, item)
            if os.path.isdir(patient_path):
                patient_folders.append(item)
    return sorted(patient_folders)


def find_images_for_patient(patient_id):
    """Find all image files for a given patient."""
    patient_path = os.path.join(BASE_DATA_PATH, patient_id)
    image_files = {}
    
    # Look for .mha files in the patient folder
    mha_files = glob.glob(os.path.join(patient_path, "*.mha"))
    
    for mha_file in mha_files:
        filename = os.path.basename(mha_file)
        
        # Skip mask files
        if "_mask.mha" in filename:
            continue
            
        # Extract image_id and modality from filename
        # Expected format: image_id_modality.mha
        base_name = filename.replace(".mha", "")
        
        # Find modality by checking suffix
        modality = None
        image_id = None
        
        for mod in MODALITIES:
            if base_name.endswith(f"_{mod}"):
                modality = mod
                image_id = base_name[:-len(f"_{mod}")]
                break
        
        if modality and image_id:
            if image_id not in image_files:
                image_files[image_id] = {}
            
            # Check if corresponding mask exists
            mask_file = os.path.join(patient_path, f"{image_id}_mask.mha")
            if os.path.exists(mask_file):
                image_files[image_id][modality] = {
                    'image_path': mha_file,
                    'mask_path': mask_file
                }
    
    return image_files


def process_single_image(image_path, mask_path, image_id, modality):
    """Process a single image and extract radiomic features for all slices."""
    try:
        # Load image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # Resample mask to match image geometry
        mask = resample_mask(mask, image)
        
        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)
        
        # Find slices with masks
        slices_with_masks = find_slices_with_masks(mask_np)
        
        if not slices_with_masks:
            print(f"[!] No slices with masks found for {image_id}_{modality}")
            return []
        
        # Extract features for each slice with mask
        all_slice_features = []
        
        for slice_idx in slices_with_masks:
            img_slice = image_np[slice_idx]
            mask_slice = mask_np[slice_idx]
            
            if np.sum(mask_slice) == 0:
                continue
            
            # Extract 2D radiomic features
            features = extract_radiomic_features_2d(img_slice, mask_slice)
            
            if features:
                # Add slice and modality information
                features_row = {
                    'slice_id': slice_idx,
                    'modality': modality,
                    'image_id': image_id,  # ← ADD THIS LINE!
                    **features
                }
                all_slice_features.append(features_row)
        
        return all_slice_features
    
    except Exception as e:
        print(f"[!] Error processing {image_id}_{modality}: {str(e)}")
        return []


def process_patient(patient_id):
    """Process all images for a single patient."""
    print(f"\n[INFO] Processing patient: {patient_id}")
    
    # Find all images for this patient
    patient_images = find_images_for_patient(patient_id)
    
    if not patient_images:
        print(f"[!] No valid images found for patient {patient_id}")
        return
    
    # Create output directory for this patient
    patient_output_dir = os.path.join(OUTPUT_BASE_PATH, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Process each image_id
    for image_id, modalities_data in patient_images.items():
        print(f"[INFO] Processing image: {image_id}")
        
        all_features_for_image = []
        
        # Process each modality for this image
        for modality, paths in modalities_data.items():
            print(f"[INFO] Processing modality: {modality}")
            
            image_path = paths['image_path']
            mask_path = paths['mask_path']
            
            # Extract features for this modality
            modality_features = process_single_image(
                image_path, mask_path, image_id, modality
            )
            
            all_features_for_image.extend(modality_features)
            print(f"[✓] Extracted features for {len(modality_features)} slices in {modality}")
        
        # Save features to CSV
        if all_features_for_image:
            output_csv = os.path.join(patient_output_dir, f"{image_id}_radiomic_features.csv")
            df = pd.DataFrame(all_features_for_image)
            df.to_csv(output_csv, index=False)
            print(f"[✓] Saved {len(all_features_for_image)} feature rows to {output_csv}")
        else:
            print(f"[!] No features extracted for image {image_id}")


def main():
    """Main function to process all patients."""
    print("[INFO] Starting multi-parametric MRI radiomic feature extraction...")
    print(f"[INFO] Base data path: {BASE_DATA_PATH}")
    print(f"[INFO] Output path: {OUTPUT_BASE_PATH}")
    print(f"[INFO] Modalities: {MODALITIES}")
    
    # Find all patient folders
    patient_folders = find_patient_folders()
    
    if not patient_folders:
        print(f"[!] No patient folders found in {BASE_DATA_PATH}")
        return
    
    print(f"[INFO] Found {len(patient_folders)} patient folders: {patient_folders}")
    
    # Process each patient with progress bar
    for patient_id in tqdm(patient_folders, desc="Processing patients"):
        try:
            process_patient(patient_id)
        except Exception as e:
            print(f"[!] Error processing patient {patient_id}: {str(e)}")
            continue
    
    print(f"\n[✓] Processing complete!")
    print(f"[INFO] Results saved to: {OUTPUT_BASE_PATH}")
    
    # Summary statistics
    total_csv_files = 0
    for patient_id in patient_folders:
        patient_output_dir = os.path.join(OUTPUT_BASE_PATH, patient_id)
        if os.path.exists(patient_output_dir):
            csv_files = glob.glob(os.path.join(patient_output_dir, "*_radiomic_features.csv"))
            total_csv_files += len(csv_files)
    
    print(f"[INFO] Total CSV files created: {total_csv_files}")


if __name__ == "__main__":
    main()