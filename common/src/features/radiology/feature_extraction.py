#!/usr/bin/env python3
"""
Feature Extraction Module

Contains the inference logic for extracting features from the model ensemble.
"""
import torch
import numpy as np
import functools
from pathlib import Path
from typing import List, Dict

# Assumes models.py is in the same package
from .models import create_network_from_params

def run_ensemble_feature_extraction(
    network_params: Dict,
    model_folder: Path,
    input_tensor: torch.Tensor,
    folds_to_use: List[int]
) -> np.ndarray:
    """
    Run ensemble inference across multiple folds to extract bottleneck features.
    """
    print("   🎯 Running ensemble feature extraction...")
    
    ensemble_features = None
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
            encoder_features = []
            def hook_fn(module, input, output):
                encoder_features.append(output)
            
            # Hook the final encoder block (bottleneck)
            hook = network.conv_blocks_context[-1].register_forward_hook(hook_fn)
            _ = network(input_tensor)
            hook.remove()
            
            bottleneck_features = encoder_features[-1]
            pooled_features = torch.nn.functional.adaptive_avg_pool3d(bottleneck_features, 1).squeeze().numpy()
            
            if ensemble_features is None:
                ensemble_features = pooled_features
            else:
                ensemble_features += pooled_features
            ensemble_count += 1
            print(f"      ✅ Extracted features from fold {fold}")
            
    if ensemble_count == 0:
        raise RuntimeError("No valid checkpoints found for feature extraction!")
    
    # Average the ensemble features
    ensemble_features /= ensemble_count
    print(f"   🎯 Ensemble feature extraction complete using {ensemble_count} folds")
    
    return ensemble_features