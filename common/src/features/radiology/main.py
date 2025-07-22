#!/usr/bin/env python3
"""
nnU-Net v1 Direct Inference and Feature Extraction Script

⚠️  IMPORTANT: This model is trained for CLINICALLY SIGNIFICANT PROSTATE CANCER DETECTION,
    not whole prostate segmentation! Low probabilities on healthy cases are expected.

Model Details:
- Task: Clinically Significant Prostate Cancer (csPCa) detection
- Input: T2W, ADC, and High b-value DWI sequences
- Output: Detection map with lesion candidate probabilities (post-processed)
- Feature extraction: Optional bottleneck features at the deepest layer

Workflow:
1. Loads input images and applies picai_prep preprocessing
2. Runs ensemble inference across all 5 folds 
3. Either extracts features from bottleneck layer OR creates detection map
4. Saves results in appropriate format

Requirements:
- nnunet, torch, simpleitk, numpy, scipy
- picai_prep (for preprocessing)
- report_guided_annotation (for post-processing)
"""
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from datetime import datetime
import os
import functools
from batchgenerators.utilities.file_and_folder_operations import load_pickle
from typing import List, Optional, Tuple
from collections import defaultdict
import tempfile
from scipy.ndimage import zoom

# --- nnU-Net v1 IMPORTS ---
# These are the core architectural components from the v1 library.
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He

# --- Preprocessing and post-processing imports ---
from picai_prep.preprocessing import Sample
from report_guided_annotation import extract_lesion_candidates
from picai_prep.preprocessing import crop_or_pad

# ============================================================================
# DATA HANDLING MODULE - Case Discovery and Data Operations
# ============================================================================
# ============================================================================
# CASE DISCOVERY FUNCTIONS
# ============================================================================
def discover_cases(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """
    Discover all valid cases in the input directory.
    Supports multiple input structures and returns a list of (case_id, file_paths).
    """
    print("--- Discovering cases ---")
    
    # First, check if we have modality-based subdirectories (current test structure)
    modality_dirs = ['axial-t2-prostate-mri', 'axial-adc-prostate-mri', 'transverse-hbv-prostate-mri']
    has_modality_structure = all((input_directory / mod_dir).exists() for mod_dir in modality_dirs)
    
    cases_to_process = []
    
    if has_modality_structure:
        print("   📋 Detected modality-based directory structure")
        cases_to_process = _discover_cases_modality_structure(input_directory, modality_dirs)
    else:
        print("   📋 Checking for case-based or direct file structure")
        cases_to_process = _discover_cases_standard_structure(input_directory)
    
    if not cases_to_process:
        print("❌ ERROR: No valid cases found to process")
        return []
        
    print(f"🎯 Ready to process {len(cases_to_process)} cases")
    return cases_to_process

def _discover_cases_modality_structure(input_directory: Path, modality_dirs: List[str]) -> List[Tuple[str, List[str]]]:
    """Discover cases in modality-based directory structure."""
    files_by_case = defaultdict(dict)
    
    for mod_dir in modality_dirs:
        modality_path = input_directory / mod_dir
        mha_files = list(modality_path.glob("*.mha"))
        
        for file_path in mha_files:
            filename = file_path.name
            if '_t2w.mha' in filename:
                case_id = filename.replace('_t2w.mha', '')
                files_by_case[case_id]['t2w'] = str(file_path)
            elif '_adc.mha' in filename:
                case_id = filename.replace('_adc.mha', '')
                files_by_case[case_id]['adc'] = str(file_path)
            elif '_hbv.mha' in filename:
                case_id = filename.replace('_hbv.mha', '')
                files_by_case[case_id]['hbv'] = str(file_path)
    
    return _validate_and_format_cases(files_by_case)

def _discover_cases_standard_structure(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """Discover cases in case-based folders or direct files."""
    case_folders = []
    direct_files = []
    
    # Look for case directories or direct files
    for item in input_directory.iterdir():
        if item.is_dir():
            case_folders.append(item)
        elif item.name.endswith(('_t2w.mha', '_adc.mha', '_hbv.mha')):
            direct_files.append(item)
    
    if case_folders:
        # Multi-folder structure (like batch script expects)
        print(f"   📋 Found {len(case_folders)} case folders")
        return _discover_cases_from_folders(case_folders)
    elif direct_files:
        # Single directory with direct files
        print("   📋 Found direct files in input directory")
        return _discover_cases_from_direct_files(direct_files)
    else:
        # Search subdirectories for files
        print("   📋 Searching subdirectories for files...")
        return _discover_cases_from_subdirs(input_directory)

def _discover_cases_from_folders(case_folders: List[Path]) -> List[Tuple[str, List[str]]]:
    """Discover cases from individual case folders."""
    cases_to_process = []
    for case_folder in sorted(case_folders):
        case_info = find_case_files(case_folder)
        if case_info is not None:
            case_id, image_files = case_info
            cases_to_process.append((case_id, image_files))
            print(f"      ✅ {case_id}: {[Path(f).name for f in image_files]}")
        else:
            print(f"      ❌ Skipped {case_folder.name}: Missing required files")
    return cases_to_process

def _discover_cases_from_direct_files(direct_files: List[Path]) -> List[Tuple[str, List[str]]]:
    """Discover cases from direct files in input directory."""
    files_by_case = defaultdict(dict)
    
    for file_path in direct_files:
        filename = file_path.name
        if '_t2w.mha' in filename:
            case_id = filename.replace('_t2w.mha', '')
            files_by_case[case_id]['t2w'] = str(file_path)
        elif '_adc.mha' in filename:
            case_id = filename.replace('_adc.mha', '')
            files_by_case[case_id]['adc'] = str(file_path)
        elif '_hbv.mha' in filename:
            case_id = filename.replace('_hbv.mha', '')
            files_by_case[case_id]['hbv'] = str(file_path)
    
    return _validate_and_format_cases(files_by_case)

def _discover_cases_from_subdirs(input_directory: Path) -> List[Tuple[str, List[str]]]:
    """Discover cases by recursively searching subdirectories."""
    all_files = []
    for subdir in input_directory.iterdir():
        if subdir.is_dir():
            all_files.extend(subdir.glob("*.mha"))
    
    files_by_case = defaultdict(dict)
    
    for file_path in all_files:
        filename = file_path.name
        if '_t2w.mha' in filename:
            case_id = filename.replace('_t2w.mha', '')
            files_by_case[case_id]['t2w'] = str(file_path)
        elif '_adc.mha' in filename:
            case_id = filename.replace('_adc.mha', '')
            files_by_case[case_id]['adc'] = str(file_path)
        elif '_hbv.mha' in filename:
            case_id = filename.replace('_hbv.mha', '')
            files_by_case[case_id]['hbv'] = str(file_path)
    
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
    """
    Find the three required MRI files in a case folder.
    Returns: (case_id, [t2w_path, adc_path, hbv_path]) or None if files not found
    """
    try:
        # Look for the three modality files
        t2w_files = list(case_folder.glob("*_t2w.mha"))
        adc_files = list(case_folder.glob("*_adc.mha"))
        hbv_files = list(case_folder.glob("*_hbv.mha"))
        
        if len(t2w_files) != 1 or len(adc_files) != 1 or len(hbv_files) != 1:
            print(f"   ⚠️ Expected exactly 1 file of each type in {case_folder.name}, found:")
            print(f"      T2W: {len(t2w_files)}, ADC: {len(adc_files)}, HBV: {len(hbv_files)}")
            return None
        
        # Extract case ID from filename (assuming format: caseID_modality.mha)
        t2w_name = t2w_files[0].stem
        case_id = t2w_name.replace("_t2w", "")
        
        return case_id, [str(t2w_files[0]), str(adc_files[0]), str(hbv_files[0])]
        
    except Exception as e:
        print(f"   ❌ Error processing folder {case_folder.name}: {e}")
        return None

def load_and_preprocess_case(case_id: str, input_files: List[str], target_patch_size: List[int]) -> Tuple[torch.Tensor, sitk.Image, tuple]:
    """
    Load and preprocess a single case for nnU-Net inference.
    
    Args:
        case_id: Case identifier
        input_files: List of image file paths [t2w, adc, hbv]
        target_patch_size: Target patch size for network input
        
    Returns:
        Tuple of (input_tensor, original_scan, original_shape)
    """
    print(f"📁 Loading input images for case: {case_id}")
    scans = [sitk.ReadImage(f, sitk.sitkFloat32) for f in input_files]
    
    print("   📊 Before preprocessing:")
    for i, scan in enumerate(scans):
        arr = sitk.GetArrayFromImage(scan)
        print(f"      Scan {i}: {arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
    
    # Apply picai_prep preprocessing (same as reference implementation)
    print("   🔧 Applying picai_prep preprocessing...")
    sample = Sample(scans=scans)
    sample.preprocess()
    
    print("   📊 After preprocessing:")
    for i, scan in enumerate(sample.scans):
        arr = sitk.GetArrayFromImage(scan)
        print(f"      Scan {i}: {arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
    
    # Store original scan for coordinate mapping
    original_scan = sample.scans[0]  # Use T2W as reference
    original_shape = sitk.GetArrayFromImage(original_scan).shape
    
    # Simple resize to target patch size (this step should match nnUNet preprocessing)
    processed_channels = []
    for i, scan in enumerate(sample.scans):
        img_array = sitk.GetArrayFromImage(scan)  # [D, H, W]
        
        # Resize to target patch size
        zoom_factors = [
            target_patch_size[0] / img_array.shape[0],
            target_patch_size[1] / img_array.shape[1], 
            target_patch_size[2] / img_array.shape[2]
        ]
        resized_array = zoom(img_array, zoom_factors, order=1)
        print(f"      Scan {i}: Resized from {img_array.shape} to {resized_array.shape}")
        
        # nnUNet-style normalization (per channel)
        # For medical images, nnUNet typically normalizes to zero mean and unit variance
        if resized_array.std() > 0:
            normalized_array = (resized_array - resized_array.mean()) / resized_array.std()
            print(f"      Scan {i}: After normalization - mean={normalized_array.mean():.6f}, std={normalized_array.std():.6f}, range=[{normalized_array.min():.3f}, {normalized_array.max():.3f}]")
        else:
            normalized_array = resized_array
            print(f"      Scan {i}: No normalization applied (std=0)")
        
        processed_channels.append(normalized_array)
    
    # Create input tensor: [1, C, D, H, W]
    input_tensor = torch.from_numpy(np.stack(processed_channels, axis=0)).unsqueeze(0).float()
    print(f"   ✅ Input tensor created with shape: {input_tensor.shape}")
    
    return input_tensor, original_scan, original_shape

def save_detection_map(detection_map: np.ndarray, original_scan: sitk.Image, 
                      case_id: str, output_folder: Path) -> None:
    """Save detection map to output folder."""
    # Create detection map in original space
    detection_sitk = sitk.GetImageFromArray(detection_map.astype(np.float32))
    detection_sitk.CopyInformation(original_scan)  # Copy spacing, origin, direction
    
    # Save detection map (this is the main output - same as reference)
    detection_output_path = output_folder / f"{case_id}.mha"
    sitk.WriteImage(detection_sitk, str(detection_output_path))
    
    print(f"      📊 Detection map: {detection_map.shape}, range: [{detection_map.min():.3f}, {detection_map.max():.3f}]")
    print(f"      📊 Non-zero voxels: {np.sum(detection_map > 0)}")
    print(f"      ✔ Saved to: {detection_output_path}")

def save_features(features: np.ndarray, case_id: str, output_folder: Path) -> None:
    """Save extracted features to output folder as .pt file."""
    output_path = output_folder / f"{case_id}_features.pt"
    # Convert numpy array to torch tensor and save
    features_tensor = torch.from_numpy(features)
    torch.save(features_tensor, output_path)
    print(f"      ✅ Feature extraction complete. Ensemble features shape: {features.shape}")
    print(f"      ✔ Features saved to: {output_path}")

# ============================================================================
# NEURAL NETWORK MODULE - Model Configuration and Operations  
# ============================================================================
def load_model_configuration(model_folder: Path) -> dict:
    """Load and validate nnU-Net model configuration."""
    print("--- Loading nnU-Net configuration ---")
    
    # Validate model folder
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")
    
    # Load plans
    plans_path = model_folder / "plans.pkl"
    if not plans_path.exists():
        raise FileNotFoundError(f"plans.pkl not found: {plans_path}")
        
    plans = load_pickle(str(plans_path))
    print("   ✅ Plans loaded successfully")
    
    # Display network architecture info
    net_params = plans['plans_per_stage'][0]
    num_input_channels = plans['num_modalities']
    num_classes = plans['num_classes'] + 1  # +1 for background
    num_pool = len(net_params['pool_op_kernel_sizes'])
    
    print("--- Network Architecture ---")
    print(f"   📊 Network parameters:")
    print(f"      - Input channels: {num_input_channels}")
    print(f"      - Base features: {plans['base_num_features']}")
    print(f"      - Num classes: {num_classes}")
    print(f"      - Num pool: {num_pool}")
    
    return plans

def get_network_parameters(plans: dict) -> dict:
    """Extract network parameters from plans."""
    net_params = plans['plans_per_stage'][0]
    
    return {
        'net_params': net_params,
        'num_input_channels': plans['num_modalities'],
        'num_classes': plans['num_classes'] + 1,  # +1 for background
        'num_pool': len(net_params['pool_op_kernel_sizes']),
        'conv_op': nn.Conv3d,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': nn.Dropout3d,
        'dropout_op_kwargs': {'p': 0, 'inplace': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True}
    }

def create_network(plans: dict, network_params: dict) -> 'Generic_UNet':
    """Create a Generic_UNet instance with the correct parameters."""
    network = Generic_UNet(
        input_channels=network_params['num_input_channels'],
        base_num_features=plans['base_num_features'],
        num_classes=network_params['num_classes'],
        num_pool=network_params['num_pool'],
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=network_params['conv_op'],
        norm_op=network_params['norm_op'],
        norm_op_kwargs=network_params['norm_op_kwargs'],
        dropout_op=network_params['dropout_op'],
        dropout_op_kwargs=network_params['dropout_op_kwargs'],
        nonlin=network_params['nonlin'],
        nonlin_kwargs=network_params['nonlin_kwargs'],
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=network_params['net_params']['pool_op_kernel_sizes'],
        conv_kernel_sizes=network_params['net_params']['conv_kernel_sizes'],
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=True
    )
    return network

def run_ensemble_inference(network_params: dict, model_folder: Path, input_tensor: torch.Tensor,
                          folds_to_use: List[int], extract_features: bool) -> np.ndarray:
    """
    Run ensemble inference across multiple folds.
    
    Args:
        network_params: Network architecture parameters
        model_folder: Path to model folder containing fold checkpoints
        input_tensor: Preprocessed input tensor
        folds_to_use: List of fold numbers to use for ensemble
        extract_features: Whether to extract features or generate segmentation
        
    Returns:
        Ensemble predictions (either features or probability map)
    """
    print("   🎯 Running ensemble inference...")
    
    pred_ensemble = None
    ensemble_count = 0
    
    # Patch torch.load to handle old checkpoints from different PyTorch versions.
    load_patched = functools.partial(torch.load, map_location=torch.device('cpu'), weights_only=False)
    
    for fold in folds_to_use:
        print(f"      🔄 Processing fold {fold}...")
        
        # Load checkpoint for this fold
        checkpoint_path = model_folder / f"fold_{fold}" / "model_best.model"
        if not checkpoint_path.is_file():
            print(f"      ⚠️ Checkpoint file not found: {checkpoint_path}, skipping fold {fold}")
            continue
            
        checkpoint = load_patched(checkpoint_path)
        state_dict = checkpoint['state_dict']
        
        # Create the network using the helper function
        network = create_network_from_params(network_params)
        
        # Load the state dict into our network instance
        network.load_state_dict(state_dict)
        network.eval()
        
        # Run inference
        with torch.no_grad():
            if extract_features:
                print(f"      🔧 Extracting encoder features from fold {fold}")
                
                # Extract encoder features using hooks
                encoder_features = []
                
                def hook_fn(module, input, output):
                    encoder_features.append(output)
                
                # Register hooks on the encoder blocks
                hooks = []
                for i, conv_block in enumerate(network.conv_blocks_context):
                    hook = conv_block.register_forward_hook(hook_fn)
                    hooks.append(hook)
                
                # Run forward pass to trigger hooks
                _ = network(input_tensor)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # Take the deepest encoder features (bottleneck)
                bottleneck_features = encoder_features[-1]
                
                # Pool the features to get a single vector representation
                fold_features = torch.nn.functional.adaptive_avg_pool3d(bottleneck_features, 1).squeeze().numpy()
                
                # Add to ensemble
                if pred_ensemble is None:
                    pred_ensemble = fold_features
                else:
                    pred_ensemble += fold_features
                ensemble_count += 1
                
            else:  # Segmentation Mode
                print(f"      🎯 Running segmentation inference for fold {fold}")
                
                # Standard inference (TTA disabled for stability)
                output_logits = network(input_tensor)
                final_logits = output_logits[0] if isinstance(output_logits, (list, tuple)) else output_logits
                probabilities = torch.nn.functional.softmax(final_logits, dim=1)
                fold_prob_map = probabilities[0, 1].cpu().numpy()  # Take foreground probability
                
                # Add to ensemble
                if pred_ensemble is None:
                    pred_ensemble = fold_prob_map
                else:
                    pred_ensemble += fold_prob_map
                ensemble_count += 1
        
        print(f"      ✅ Fold {fold} completed")
    
    if ensemble_count == 0:
        raise RuntimeError("No valid checkpoints found!")
    
    # Average the ensemble predictions
    pred_ensemble /= ensemble_count
    print(f"   🎯 Ensemble complete using {ensemble_count} folds")
    
    return pred_ensemble

def create_network_from_params(network_params: dict) -> 'Generic_UNet':
    """Create a Generic_UNet instance from network parameters dictionary."""
    return Generic_UNet(
        input_channels=network_params['num_input_channels'],
        base_num_features=network_params['base_num_features'],
        num_classes=network_params['num_classes'],
        num_pool=network_params['num_pool'],
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=network_params['conv_op'],
        norm_op=network_params['norm_op'],
        norm_op_kwargs=network_params['norm_op_kwargs'],
        dropout_op=network_params['dropout_op'],
        dropout_op_kwargs=network_params['dropout_op_kwargs'],
        nonlin=network_params['nonlin'],
        nonlin_kwargs=network_params['nonlin_kwargs'],
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=network_params['net_params']['pool_op_kernel_sizes'],
        conv_kernel_sizes=network_params['net_params']['conv_kernel_sizes'],
        upscale_logits=False,
        convolutional_pooling=False,
        convolutional_upsampling=True
    )

def postprocess_detection_map(prob_map: np.ndarray, original_shape: tuple, probability_threshold: float = 0.35) -> np.ndarray:
    """Apply post-processing to create detection map from probability map."""
    print("      🔄 Mapping back to original image space...")
    zoom_factors_back = [
        original_shape[0] / prob_map.shape[0],
        original_shape[1] / prob_map.shape[1], 
        original_shape[2] / prob_map.shape[2]
    ]
    prob_map_original = zoom(prob_map, zoom_factors_back, order=1)
    print(f"      📐 Mapped probability shape: {prob_map_original.shape}")
    
    # Apply post-processing to create detection map
    print("      🔧 Applying lesion candidate extraction...")
    
    def extract_lesion_candidates_cropped(pred: np.ndarray, threshold):
        """Same function as in reference implementation"""
        size = pred.shape
        pred = crop_or_pad(pred, (20, 384, 384))
        pred = crop_or_pad(pred, size)
        return extract_lesion_candidates(
            pred, 
            threshold=probability_threshold,  # Threshold for lesion candidate extraction
            num_lesions_to_extract=1  # Extract only the main lesion
        )[0]
    
    detection_map = extract_lesion_candidates_cropped(prob_map_original, threshold="dynamic")
    return detection_map

# ============================================================================
# CASE PROCESSING MODULE - Main Processing Logic
# ============================================================================

def process_single_case(case_id: str, input_files: List[str], plans: dict, 
                       network_params: dict, model_folder: Path, output_folder: Path, 
                       extract_features: bool, folds_to_use: List[int], 
                       probability_threshold: float = 0.35) -> bool:
    """
    Process a single case for either feature extraction or detection map generation.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"📋 Processing case: {case_id}")
        
        # --- DATA HANDLING: Load and preprocess the case ---
        target_patch_size = plans['plans_per_stage'][0]['patch_size']  # [16, 320, 320]
        print(f"   🎯 Target patch size: {target_patch_size}")
        
        input_tensor, original_scan, original_shape = load_and_preprocess_case(
            case_id, input_files, target_patch_size
        )
        
        # Add base_num_features to network_params for network creation
        network_params['base_num_features'] = plans['base_num_features']
        
        # --- NEURAL NETWORK: Run ensemble inference ---
        ensemble_result = run_ensemble_inference(
            network_params, model_folder, input_tensor, folds_to_use, extract_features
        )

        # --- DATA HANDLING: Post-process and save output ---
        print("   💾 Saving output...")
        
        if extract_features:
            save_features(ensemble_result, case_id, output_folder)
        else:  # Segmentation Mode
            print(f"      📊 Ensemble probability map shape: {ensemble_result.shape}")
            print(f"      📊 Ensemble probability range: [{ensemble_result.min():.4f}, {ensemble_result.max():.4f}]")
            
            detection_map = postprocess_detection_map(ensemble_result, original_shape, probability_threshold)
            save_detection_map(detection_map, original_scan, case_id, output_folder)

        print(f"   ✅ Case {case_id} completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing case {case_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# MAIN EXECUTION MODULE - Orchestrates the Entire Pipeline
# ============================================================================
def main():
    """
    Main execution function with clear separation of concerns.
    
    This function orchestrates the entire pipeline by:
    1. Setting up configuration and validating paths
    2. Loading model configuration and network parameters (NEURAL NETWORK MODULE)
    3. Discovering and validating input cases (DATA HANDLING MODULE)
    4. Processing each case through the pipeline (CASE PROCESSING MODULE)
    5. Providing final summary of results
    """
    print("=" * 80)
    print("🧠 nnU-Net v1 Inference & Feature Extraction Script (Batch Processing)")
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ============================================================================
    # CONFIGURATION - EDIT THESE PATHS
    # ============================================================================
    
    # --- Mode Selection ---
    EXTRACT_FEATURES = True # Set to True for features, False for detection map
    PROBABILITY_THRESHOLD = 0.35  # Threshold for lesion candidate extraction
    # --- Input/Output Configuration ---
    INPUT_DIRECTORY = "/path/to/images"
    OUTPUT_DIRECTORY = "/path/to/output"
    MODEL_FOLDER = "/path/to/model"
    FOLDS_TO_USE = [0, 1, 2, 3, 4]

    # ============================================================================
    # 1. SETUP AND VALIDATION
    # ============================================================================
    
    # Convert to Path objects and validate
    input_directory = Path(INPUT_DIRECTORY)
    output_directory = Path(OUTPUT_DIRECTORY) 
    model_folder = Path(MODEL_FOLDER)

    if not input_directory.exists():
        print(f"❌ ERROR: Input directory does not exist: {input_directory}")
        return
    if not model_folder.exists():
        print(f"❌ ERROR: Model folder does not exist: {model_folder}")
        return

    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Input directory: {input_directory}")
    print(f"📁 Output directory: {output_directory}")
    print(f"🧠 Model folder: {model_folder}")
    print(f"🎯 Mode: {'Feature Extraction' if EXTRACT_FEATURES else 'Detection Map Generation'}")

    # ============================================================================
    # 2. LOAD MODEL CONFIGURATION
    # ============================================================================
    
    try:
        plans = load_model_configuration(model_folder)
        network_params = get_network_parameters(plans)
    except Exception as e:
        print(f"❌ ERROR: Failed to load model configuration: {e}")
        return

    # ============================================================================
    # 3. DISCOVER CASES
    # ============================================================================
    
    cases_to_process = discover_cases(input_directory)
    if not cases_to_process:
        return

    # ============================================================================
    # 4. BATCH PROCESSING
    # ============================================================================
    
    print("=" * 80)
    print("🚀 STARTING BATCH PROCESSING")
    print("=" * 80)
    
    successful_cases = 0
    failed_cases = 0

    for i, (case_id, image_files) in enumerate(cases_to_process, 1):
        print(f"\n[{i}/{len(cases_to_process)}] " + "="*60)
        
        success = process_single_case(
            case_id=case_id,
            input_files=image_files,
            plans=plans,
            network_params=network_params,
            model_folder=model_folder,
            output_folder=output_directory,
            extract_features=EXTRACT_FEATURES,
            folds_to_use=FOLDS_TO_USE,
            probability_threshold=PROBABILITY_THRESHOLD
        )
        
        if success:
            successful_cases += 1
        else:
            failed_cases += 1

    # ============================================================================
    # 5. FINAL SUMMARY
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("📊 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"✅ Successful cases: {successful_cases}")
    print(f"❌ Failed cases: {failed_cases}")
    print(f"📁 Output directory: {output_directory}")
    if EXTRACT_FEATURES:
        print(f"🧠 Features saved as: {output_directory}/*_features.pt")
    else:
        print(f"🎯 Detection maps saved as: {output_directory}/*.mha")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
