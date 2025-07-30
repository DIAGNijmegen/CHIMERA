import pickle
import os
import argparse
import sys

from pathlib import Path

# Add the project's base directory to the Python path so we can import our custom modules
# This goes up four directories to the project root (CHIMERA).
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from task1_baseline.prediction_model.Aggregators.wsi_datasets.clinical_processor import ClinicalDataProcessor

def main(args):
    """
    This script creates and saves a 'fitted' ClinicalDataProcessor.
    It should be run once after training, using the same clinical data
    that the model was trained on.
    """
    print(f"Scanning for all clinical JSON files in: {args.clinical_data_dir}")
    
    # Discover all case IDs from the filenames in your training clinical data directory
    all_case_ids = [f.replace('.json', '') for f in os.listdir(args.clinical_data_dir) if f.endswith('.json')]
    
    if not all_case_ids:
        raise ValueError(f"No clinical data files (.json) found in the specified directory: {args.clinical_data_dir}")
        
    print(f"Found {len(all_case_ids)} cases. Fitting the clinical processor...")
    
    # 1. Initialize the processor
    clinical_processor = ClinicalDataProcessor(clinical_data_path=args.clinical_data_dir)
    
    # 2. Fit it on all available training data to learn the normalization/encoding rules
    clinical_processor.fit(all_case_ids)
    
    # 3. Save the fitted processor object to a .pkl file
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, 'clinical_processor.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(clinical_processor, f)
        
    print(f"\n✅ Successfully created and saved 'clinical_processor.pkl' to '{save_path}'")
    print("You can now copy this file to your 'model' directory for inference.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit and save a ClinicalDataProcessor.")
    parser.add_argument('--clinical_data_dir', type=str, required=True, help='Path to the directory containing ALL clinical JSON files from your training set.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the clinical_processor.pkl file will be saved.')
    
    # Example usage:
    # python create_processor.py --clinical_data_dir /path/to/training/clinical_data --output_dir ./model_assets
    
    args = parser.parse_args()
    main(args)
