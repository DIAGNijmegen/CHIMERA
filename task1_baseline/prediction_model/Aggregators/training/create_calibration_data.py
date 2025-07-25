# fit_exponential_trend.py

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from tqdm import tqdm

# --- Add Project Root to Python Path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
aggregators_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, aggregators_path)

from Aggregators.mil_models import create_downstream_model
from Aggregators.wsi_datasets.clinical_processor import ClinicalDataProcessor

def load_all_features_for_case(case_id, path_dir, rad_dir, clinical_processor):
    """Loads all pre-computed features for a single case."""
    pathology_feature_files = glob.glob(str(path_dir / f"{case_id}_*.pt"))
    if not pathology_feature_files: raise FileNotFoundError(f"No pathology features for {case_id}")
    all_path_features = [torch.load(fp, map_location='cpu') for fp in sorted(pathology_feature_files)]
    pathology_features_tensor = torch.cat(all_path_features, dim=0).unsqueeze(0)

    radiology_feature_file = rad_dir / f"{case_id}_0001_features.pt"
    if not radiology_feature_file.exists():
        return pathology_features_tensor, torch.zeros(1, 320), clinical_processor.transform(case_id).unsqueeze(0)
    
    radiology_features_tensor = torch.load(radiology_feature_file, map_location='cpu')
    if len(radiology_features_tensor.shape) == 1: radiology_features_tensor = radiology_features_tensor.unsqueeze(0)
    
    clinical_features_tensor = clinical_processor.transform(case_id).unsqueeze(0)
    return pathology_features_tensor, radiology_features_tensor, clinical_features_tensor

def exponential_func(x, a, b):
    """Defines an exponential decay function."""
    return a * np.exp(-b * x)

def main(args):
    """
    Fits both linear and exponential trend lines to event data and saves the parameters for both.
    """
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load Model and Assets ---
    # (This section is unchanged)
    print("--- Step 1: Loading trained model and assets ---")
    model_dir = Path(args.model_dir)
    checkpoint = torch.load(model_dir / "s_checkpoint.pth", map_location=torch.device('cpu'))
    with open(model_dir / "clinical_processor.pkl", 'rb') as f: clinical_processor = pickle.load(f)
    if 'config' in checkpoint:
        config_dict, saved_weights = checkpoint['config'], checkpoint['model']
    else:
        with open(model_dir / "config.json", 'r') as f: config_dict = json.load(f)
        saved_weights = checkpoint
    config_args = argparse.Namespace(**config_dict)
    config_args.clinical_processor = clinical_processor
    model = create_downstream_model(args=config_args, mode='survival')
    model.load_state_dict(saved_weights)
    model.eval()

    # --- Step 2: Get Predictions and Ground Truth ---
    # (This section is unchanged)
    print("\n--- Step 2: Getting risk predictions using real features ---")
    clinical_dir, path_dir, rad_dir = Path(args.clinical_data_dir), Path(args.pathology_features_dir), Path(args.radiology_features_dir)
    all_case_ids = [p.stem for p in clinical_dir.glob("*.json")]
    predictions = []
    for case_id in tqdm(all_case_ids, desc="Processing cases"):
        try:
            json_path = clinical_dir / f"{case_id}.json"
            with open(json_path, 'r') as f: data_from_json = json.load(f)
            clinical_data = data_from_json[0] if isinstance(data_from_json, list) else data_from_json
            path_feats, rad_feats, clin_feats = load_all_features_for_case(case_id, path_dir, rad_dir, clinical_processor)
            with torch.no_grad():
                output_dict = model.forward_no_loss(h=path_feats, additional_embeddings=rad_feats, clinical_features=clin_feats)
            log_risk = output_dict['logits'].item()
            predictions.append({'case_id': case_id, 'time': clinical_data.get("time_to_follow-up/BCR"), 'event_observed': int(float(clinical_data.get("BCR", 0.0))), 'log_risk': log_risk})
        except Exception as e:
            print(f"Could not process case {case_id}. Skipping. Error: {e}")

    pred_df = pd.DataFrame(predictions).dropna()
    pred_df['risk_score'] = np.exp(pred_df['log_risk'])
    event_df = pred_df[pred_df['event_observed'] == 1].copy()
    censored_df = pred_df[pred_df['event_observed'] == 0].copy()

    # --- Step 3: Fit Linear and Exponential Models to Event Data ---
    print("\n--- Step 3: Fitting models to event data (Time vs. Risk Score) ---")
    if len(event_df) < 2:
        raise ValueError("Not enough data points with events (<2) to fit models.")
    
    x_data = event_df['risk_score']
    y_data = event_df['time']

    # Fit a linear model (degree 1 polynomial)
    slope_lin, intercept_lin = np.polyfit(x_data, y_data, 1)
    print(f"Linear fit complete. Slope: {slope_lin:.4f}, Intercept: {intercept_lin:.4f}")

    # Fit an exponential model
    # Provide initial guesses for parameters a and b to help the solver
    initial_guess = [y_data.max(), 1.0] 
    params_exp, _ = curve_fit(exponential_func, x_data, y_data, p0=initial_guess, maxfev=5000)
    a_exp, b_exp = params_exp
    print(f"Exponential fit complete. Coeffs (a,b): a={a_exp:.4f}, b={b_exp:.4f}")

    # --- Step 4: Save Both Sets of Calibration Parameters ---
    print("\n--- Step 4: Saving calibration parameter files ---")
    # Save linear model
    calib_linear = {'type': 'linear_risk_to_time', 'slope': slope_lin, 'intercept': intercept_lin}
    with open(output_dir / 'linear_event_calibration.pkl', 'wb') as f:
        pickle.dump(calib_linear, f)
    print(f"✅ Linear calibration parameters saved.")

    # Save exponential model
    calib_exp = {'type': 'exp_risk_to_time', 'a': a_exp, 'b': b_exp}
    with open(output_dir / 'exponential_event_calibration.pkl', 'wb') as f:
        pickle.dump(calib_exp, f)
    print(f"✅ Exponential calibration parameters saved.")

    # --- Step 5: Create and Save Visualization with Both Trend Lines ---
    print("\n--- Step 5: Creating and saving the visualization ---")
    plt.figure(figsize=(12, 8))
    plt.scatter(censored_df['risk_score'], censored_df['time'], alpha=0.6, marker='x', s=40, color='orange', label='Censored (No BCR)')
    plt.scatter(event_df['risk_score'], event_df['time'], alpha=0.7, marker='o', s=50, edgecolor='k', color='royalblue', label='BCR Event')
    
    line_x = np.linspace(pred_df['risk_score'].min(), pred_df['risk_score'].max(), 200)
    
    # Plot linear trend line
    line_y_lin = slope_lin * line_x + intercept_lin
    plt.plot(line_x, line_y_lin, color='red', linewidth=2, linestyle='--', label='Linear Fit')

    # Plot exponential trend line
    line_y_exp = exponential_func(line_x, a_exp, b_exp)
    plt.plot(line_x, line_y_exp, color='green', linewidth=2.5, label='Exponential Fit')
    
    plt.title('Linear vs. Exponential Fit for Event Data')
    plt.xlabel('Predicted Risk Score (Hazard Ratio)')
    plt.ylabel('Actual Time (Months)')
    plt.ylim(bottom=-5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    output_path_png = output_dir / 'exp_vs_linear_fit_comparison.png'
    plt.savefig(output_path_png, dpi=300)
    print(f"✅ Plot with both trend lines successfully saved to {output_path_png}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fit linear and exponential trend lines to event data for calibration.")
    parser.add_argument('--pathology_features_dir', type=str, required=True)
    parser.add_argument('--radiology_features_dir', type=str, required=True)
    parser.add_argument('--clinical_data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)