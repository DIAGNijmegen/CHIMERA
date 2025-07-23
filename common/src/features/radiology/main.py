#!/usr/bin/env python3
"""
nnU-Net v1 Main Inference Script

Orchestrates the csPCa detection pipeline by calling modular components
for data handling, model creation, and processing.
"""
from pathlib import Path
from datetime import datetime

# Import from our modular scripts
from .dataset import discover_cases
from .models import load_model_configuration, get_network_parameters
from .processing import process_single_case

def main():
    """Main execution function to run the inference pipeline."""
    print("=" * 80)
    print("🧠 nnU-Net v1 Modular Inference & Feature Extraction Script")
    print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    EXTRACT_FEATURES = True
    PROBABILITY_THRESHOLD = 0.35
    INPUT_DIRECTORY = "/data/pa_cpgarchive/projects/chimera/prostate/test_data/input/interf0"
    OUTPUT_DIRECTORY = "/data/temporary/chimera/prostate/inference_output/radiology"
    MODEL_FOLDER = "/data/pa_cpgarchive/projects/chimera/algorithms/baseline/weights/pi-cai_nnUnet"
    FOLDS_TO_USE = [0, 1, 2, 3, 4]

    # ============================================================================
    # SETUP AND VALIDATION
    # ============================================================================
    input_dir = Path(INPUT_DIRECTORY)
    output_dir = Path(OUTPUT_DIRECTORY) 
    model_dir = Path(MODEL_FOLDER)

    if not input_dir.exists():
        print(f"❌ ERROR: Input directory does not exist: {input_dir}")
        return
    if not model_dir.exists():
        print(f"❌ ERROR: Model folder does not exist: {model_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"🎯 Mode: {'Feature Extraction' if EXTRACT_FEATURES else 'Detection Map Generation'}")

    # ============================================================================
    # LOAD MODEL AND DISCOVER CASES
    # ============================================================================
    try:
        plans = load_model_configuration(model_dir)
        network_params = get_network_parameters(plans)
    except Exception as e:
        print(f"❌ ERROR: Failed to load model configuration: {e}")
        return

    cases_to_process = discover_cases(input_dir)
    if not cases_to_process:
        return

    # ============================================================================
    # BATCH PROCESSING
    # ============================================================================
    print("\n" + "=" * 80)
    print("🚀 STARTING BATCH PROCESSING")
    print("=" * 80)
    
    successful_cases, failed_cases = 0, 0
    total_cases = len(cases_to_process)

    for i, (case_id, image_files) in enumerate(cases_to_process, 1):
        print(f"\n[{i}/{total_cases}] " + "="*60)
        
        success = process_single_case(
            case_id=case_id,
            input_files=image_files,
            plans=plans,
            network_params=network_params,
            model_folder=model_dir,
            output_folder=output_dir,
            extract_features=EXTRACT_FEATURES,
            folds_to_use=FOLDS_TO_USE,
            probability_threshold=PROBABILITY_THRESHOLD
        )
        
        if success:
            successful_cases += 1
        else:
            failed_cases += 1

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("📊 PROCESSING SUMMARY")
    print("=" * 80)
    print(f"✅ Successful cases: {successful_cases}")
    print(f"❌ Failed cases: {failed_cases}")
    print(f"📁 Output saved in: {output_dir}")
    print(f"Script finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()