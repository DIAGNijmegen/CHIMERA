#!/usr/bin/env python3
"""
Data Handling Module

Handles case discovery, loading, preprocessing, and saving of results.
Now supports grand-challenge.org `inputs.json` format.
"""
import torch
import numpy as np
import SimpleITK as sitk
import json
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
from scipy.ndimage import zoom

# --- Preprocessing and post-processing imports ---
from picai_prep.preprocessing import Sample, crop_or_pad
from report_guided_annotation import extract_lesion_candidates

# ============================================================================
# CASE DISCOVERY FUNCTIONS
# ============================================================================

def _discover_case_from_json(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """
    Discover a single case by using the relative_path from inputs.json to find
    the correct sub-directory and then locating the first .mha file within it.
    This version completely ignores the filename from the JSON.
    """
    json_path = input_directory / 'inputs.json'
    if not json_path.is_file():
        return []

    print("   📋 Detected grand-challenge `inputs.json` structure")
    
    slug_map = {
        "axial-t2-prostate-mri": "t2w",
        "axial-adc-prostate-mri": "adc",
        "transverse-hbv-prostate-mri": "hbv"
    }
    found_paths = {}

    with open(json_path, 'r') as f:
        inputs_data = json.load(f)

    for item in inputs_data:
        interface = item.get('interface', {})
        slug = interface.get('slug')
        
        if slug in slug_map:
            modality_key = slug_map[slug]
            relative_path = interface.get('relative_path')

            if not relative_path:
                continue

            # Construct the full path to the modality's sub-directory
            modality_dir = input_directory / relative_path
            
            if modality_dir.is_dir():
                # Find the first .mha file in that directory
                mha_files = list(modality_dir.glob("*.mha"))
                if mha_files:
                    found_paths[modality_key] = str(mha_files[0])
                else:
                    print(f"      ⚠️ Found directory for '{slug}' but no .mha file inside: {modality_dir}")
            else:
                print(f"      ⚠️ Could not find directory specified in JSON: {modality_dir}")

    # Validate that we found all required modalities
    if all(key in slug_map.values() for key in found_paths):
        case_id = input_directory.name
        # Ensure the order is correct: t2w, adc, hbv
        image_files = [found_paths['t2w'], found_paths['adc'], found_paths['hbv']]
        print(f"      ✅ Found case '{case_id}' with files: {[Path(f).name for f in image_files]}")
        return [(case_id, image_files)]
    else:
        missing = [key for key in slug_map.values() if key not in found_paths]
        print(f"      ❌ Skipped case in {input_directory.name}: Missing modalities {missing}")
        return []

def discover_cases(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """
    Discover all valid cases in the input directory.
    Supports grand-challenge.org JSON format and standard file structures.
    """
    print("--- Discovering cases ---")
    
    # 1. Prioritize Grand-Challenge JSON structure
    cases_to_process = _discover_case_from_json(input_directory)
    if cases_to_process:
        print(f"🎯 Ready to process {len(cases_to_process)} case from inputs.json")
        return cases_to_process

    # 2. Fallback to modality-based subdirectories
    print("   Could not find `inputs.json`, falling back to file system scan.")
    modality_dirs = ['axial-t2-prostate-mri', 'axial-adc-prostate-mri', 'transverse-hbv-prostate-mri']
    has_modality_structure = all((input_directory / mod_dir).exists() for mod_dir in modality_dirs)
    if has_modality_structure:
        print("   📋 Detected modality-based directory structure")
        cases_to_process = _discover_cases_modality_structure(input_directory, modality_dirs)
    else:
        # 3. Fallback to case-based or direct file structure
        print("   📋 Checking for case-based or direct file structure")
        cases_to_process = _discover_cases_standard_structure(input_directory)
    
    if not cases_to_process:
        print("❌ ERROR: No valid cases found to process")
        return []
        
    print(f"🎯 Ready to process {len(cases_to_process)} cases from file system scan")
    return cases_to_process


def _discover_cases_modality_structure(input_directory: Path, modality_dirs: List[str]) -> List[Tuple[str, List[str]]]:
    """Discover cases in modality-based directory structure."""
    files_by_case = defaultdict(dict)
    
    for mod_dir in modality_dirs:
        modality_path = input_directory / mod_dir
        for file_path in modality_path.glob("*.mha"):
            filename = file_path.name
            if '_t2w.mha' in filename:
                files_by_case[filename.replace('_t2w.mha', '')]['t2w'] = str(file_path)
            elif '_adc.mha' in filename:
                files_by_case[filename.replace('_adc.mha', '')]['adc'] = str(file_path)
            elif '_hbv.mha' in filename:
                files_by_case[filename.replace('_hbv.mha', '')]['hbv'] = str(file_path)
    
    return _validate_and_format_cases(files_by_case)

def _discover_cases_standard_structure(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """Discover cases in case-based folders or direct files."""
    case_folders = [item for item in input_directory.iterdir() if item.is_dir()]
    direct_files = [item for item in input_directory.iterdir() if item.name.endswith(('_t2w.mha', '_adc.mha', '_hbv.mha'))]
    
    if case_folders:
        print(f"   📋 Found {len(case_folders)} case folders")
        return _discover_cases_from_folders(case_folders)
    elif direct_files:
        print("   📋 Found direct files in input directory")
        return _discover_cases_from_direct_files(direct_files)
    else:
        print("   📋 Searching subdirectories for files...")
        return _discover_cases_from_subdirs(input_directory)

def _discover_cases_from_folders(case_folders: List[Path]) -> List[Tuple[str, List[str]]]:
    """Discover cases from individual case folders."""
    cases_to_process = []
    for case_folder in sorted(case_folders):
        case_info = find_case_files(case_folder)
        if case_info:
            cases_to_process.append(case_info)
            print(f"      ✅ {case_info[0]}: {[Path(f).name for f in case_info[1]]}")
        else:
            print(f"      ❌ Skipped {case_folder.name}: Missing required files")
    return cases_to_process

def _discover_cases_from_direct_files(direct_files: List[Path]) -> List[Tuple[str, List[str]]]:
    """Discover cases from direct files in input directory."""
    files_by_case = defaultdict(dict)
    for file_path in direct_files:
        filename = file_path.name
        if '_t2w.mha' in filename:
            files_by_case[filename.replace('_t2w.mha', '')]['t2w'] = str(file_path)
        elif '_adc.mha' in filename:
            files_by_case[filename.replace('_adc.mha', '')]['adc'] = str(file_path)
        elif '_hbv.mha' in filename:
            files_by_case[filename.replace('_hbv.mha', '')]['hbv'] = str(file_path)
    return _validate_and_format_cases(files_by_case)

def _discover_cases_from_subdirs(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """Discover cases by recursively searching subdirectories."""
    all_files = [f for subdir in input_directory.iterdir() if subdir.is_dir() for f in subdir.glob("*.mha")]
    files_by_case = defaultdict(dict)
    for file_path in all_files:
        filename = file_path.name
        if '_t2w.mha' in filename:
            files_by_case[filename.replace('_t2w.mha', '')]['t2w'] = str(file_path)
        elif '_adc.mha' in filename:
            files_by_case[filename.replace('_adc.mha', '')]['adc'] = str(file_path)
        elif '_hbv.mha' in filename:
            files_by_case[filename.replace('_hbv.mha', '')]['hbv'] = str(file_path)
    return _validate_and_format_cases(files_by_case)

def _validate_and_format_cases(files_by_case: dict) -> List[Tuple[str, List[str]]]:
    """Validate and format discovered cases."""
    cases_to_process = []
    for case_id, files in files_by_case.items():
        if 't2w' in files and 'adc' in files and 'hbv' in files:
            image_files = [files['t2w'], files['adc'], files['hbv']]
            cases_to_process.append((case_id, image_files))
            print(f"      ✅ {case_id}: {[Path(f).name for f in image_files]}")
        else:
            missing = [m for m in ['t2w', 'adc', 'hbv'] if m not in files]
            print(f"      ❌ Skipped {case_id}: Missing {missing}")
    return cases_to_process

def find_case_files(case_folder: Path) -> Optional[Tuple[str, List[str]]]:
    """Find the three required MRI files in a case folder."""
    try:
        t2w_files = list(case_folder.glob("*_t2w.mha"))
        adc_files = list(case_folder.glob("*_adc.mha"))
        hbv_files = list(case_folder.glob("*_hbv.mha"))
        
        if not (len(t2w_files) == 1 and len(adc_files) == 1 and len(hbv_files) == 1):
            return None
        
        case_id = t2w_files[0].stem.replace("_t2w", "")
        return case_id, [str(t2w_files[0]), str(adc_files[0]), str(hbv_files[0])]
    except Exception as e:
        print(f"   ❌ Error processing folder {case_folder.name}: {e}")
        return None

# ============================================================================
# DATA PREPROCESSING AND SAVING
# ============================================================================

def load_and_preprocess_case(case_id: str, input_files: List[str], target_patch_size: List[int]) -> Tuple[torch.Tensor, sitk.Image, tuple]:
    """Load and preprocess a single case for nnU-Net inference."""
    print(f"📁 Loading and preprocessing case: {case_id}")
    scans = [sitk.ReadImage(f, sitk.sitkFloat32) for f in input_files]
    
    # Apply picai_prep preprocessing
    sample = Sample(scans=scans)
    sample.preprocess()
    
    original_scan = sample.scans[0]
    original_shape = sitk.GetArrayFromImage(original_scan).shape
    
    processed_channels = []
    for scan in sample.scans:
        img_array = sitk.GetArrayFromImage(scan)
        zoom_factors = [t / s for t, s in zip(target_patch_size, img_array.shape)]
        resized_array = zoom(img_array, zoom_factors, order=1)
        
        # nnU-Net style normalization
        mean, std = resized_array.mean(), resized_array.std()
        normalized_array = (resized_array - mean) / std if std > 0 else resized_array
        processed_channels.append(normalized_array)
    
    input_tensor = torch.from_numpy(np.stack(processed_channels, axis=0)).unsqueeze(0).float()
    print(f"   ✅ Input tensor created with shape: {input_tensor.shape}")
    
    return input_tensor, original_scan, original_shape

def save_detection_map(detection_map: np.ndarray, original_scan: sitk.Image, case_id: str, output_folder: Path) -> None:
    """Save detection map to output folder."""
    output_path = output_folder / f"{case_id}.mha"
    
    # Ensure the output directory for the MHA file exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    detection_sitk = sitk.GetImageFromArray(detection_map.astype(np.float32))
    detection_sitk.CopyInformation(original_scan)
    
    sitk.WriteImage(detection_sitk, str(output_path))
    print(f"      ✔ Detection map saved to: {output_path}")

def save_features(features: np.ndarray, case_id: str, output_folder: Path) -> None:
    """Save extracted features to output folder as .pt file."""
    output_path = output_folder / f"{case_id}_features.pt"
    
    # Ensure the output directory for the PT file exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(torch.from_numpy(features), output_path)
    print(f"      ✔ Features saved to: {output_path}")

def postprocess_detection_map(prob_map: np.ndarray, original_shape: tuple, probability_threshold: float = 0.35) -> np.ndarray:
    """Apply post-processing to create detection map from probability map."""
    print("      🔄 Mapping back to original image space...")
    zoom_factors_back = [o / p for o, p in zip(original_shape, prob_map.shape)]
    prob_map_original = zoom(prob_map, zoom_factors_back, order=1)
    
    print("      🔧 Applying lesion candidate extraction...")
    
    def extract_lesion_candidates_cropped(pred: np.ndarray, threshold):
        size = pred.shape
        # This padding logic is from the original PI-CAI baseline
        pred_padded = crop_or_pad(pred, (20, 384, 384))
        pred_cropped = crop_or_pad(pred_padded, size)
        return extract_lesion_candidates(pred_cropped, threshold=probability_threshold, num_lesions_to_extract=1)[0]
    
    detection_map = extract_lesion_candidates_cropped(prob_map_original, threshold="dynamic")
    return detection_map