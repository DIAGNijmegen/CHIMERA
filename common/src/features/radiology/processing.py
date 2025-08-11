#!/usr/bin/env python3
"""
Processing Module

Contains the core logic for running segmentation inference and case processing.
"""
import torch
import numpy as np
import functools
from pathlib import Path
from typing import List, Dict, Any

# Assumes models.py, feature_extraction.py, and dataset.py are in the same package
from .models import create_network_from_params
from .feature_extraction import run_ensemble_feature_extraction
from .dataset import (
    load_and_preprocess_case,
    postprocess_detection_map,
    save_features,
    save_detection_map
)

def run_ensemble_segmentation(
    network_params: Dict,
    model_folder: Path,
    input_tensor: torch.Tensor,
    folds_to_use: List[int]
) -> np.ndarray:
    """
    Run ensemble inference across multiple folds for segmentation.
    """
    print("   🎯 Running ensemble segmentation...")
    
    pred_ensemble = None
    ensemble_count = 0
    
    load_patched = functools.partial(torch.load, map_location=torch.device('cpu'), weights_only=False)
    
    for fold in folds_to_use:
        checkpoint_path = model_folder / f"fold_{fold}" / "model_best.model"
        if not checkpoint_path.is_file():
            print(f"      ⚠️ Checkpoint not found for fold {fold}, skipping.")
            continue
            
        checkpoint = load_patched(checkpoint_path)
        network = create_network_from_params(network_params)
        network.load_state_dict(checkpoint['state_dict'])
        network.eval()
        
        with torch.no_grad():
            output_logits = network(input_tensor)
            final_logits = output_logits[0] if isinstance(output_logits, (list, tuple)) else output_logits
            probabilities = torch.nn.functional.softmax(final_logits, dim=1)
            fold_prob_map = probabilities[0, 1].cpu().numpy()
            
            if pred_ensemble is None:
                pred_ensemble = fold_prob_map
            else:
                pred_ensemble += fold_prob_map
            ensemble_count += 1
            print(f"      ✅ Segmentation complete for fold {fold}")

    if ensemble_count == 0:
        raise RuntimeError("No valid checkpoints found for segmentation!")
        
    pred_ensemble /= ensemble_count
    print(f"   🎯 Ensemble segmentation complete using {ensemble_count} folds")
    return pred_ensemble

def process_single_case(
    case_id: str,
    input_files: List[str],
    plans: Dict[str, Any],
    network_params: Dict[str, Any],
    model_folder: Path,
    output_folder: Path,
    extract_features: bool,
    folds_to_use: List[int],
    probability_threshold: float
) -> bool:
    """
    Process a single case for either feature extraction or detection map generation.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"📋 Processing case: {case_id}")
        
        target_patch_size = plans['plans_per_stage'][0]['patch_size']
        input_tensor, original_scan, original_shape = load_and_preprocess_case(
            case_id, input_files, target_patch_size
        )
        
        print("   💾 Processing and saving output...")
        if extract_features:
            ensemble_result = run_ensemble_feature_extraction(
                network_params, model_folder, input_tensor, folds_to_use
            )
            save_features(ensemble_result, case_id, output_folder)
        else:
            prob_map = run_ensemble_segmentation(
                network_params, model_folder, input_tensor, folds_to_use
            )
            detection_map = postprocess_detection_map(prob_map, original_shape, probability_threshold)
            save_detection_map(detection_map, original_scan, case_id, output_folder)

        print(f"   ✅ Case {case_id} completed successfully!")
        return True
        
    except Exception as e:
        import traceback
        print(f"   ❌ Error processing case {case_id}: {e}")
        traceback.print_exc()
        return False