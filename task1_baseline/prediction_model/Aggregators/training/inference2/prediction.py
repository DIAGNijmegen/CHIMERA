

import torch
import numpy as np

def run_inference_and_calibrate(model, pathology_features, radiology_features, clinical_features, calibration_data):
    """
    Runs model inference and calibrates the risk score to a time-to-event prediction.

    Args:
        model: The pre-trained model.
        pathology_features: Pathology features tensor.
        radiology_features: Radiology features tensor.
        clinical_features: Clinical features tensor.
        calibration_data: The loaded calibration data.

    Returns:
        The predicted time to event in months.
    """
    # --- Step 3: Perform Inference ---
    print("\n--- Step 3: Running Model Inference ---")
    with torch.no_grad():
        output_dict = model.forward_no_loss(
            h=pathology_features,
            additional_embeddings=radiology_features,
            clinical_features=clinical_features
        )
    
    logits = output_dict['logits']
    risk_score = torch.exp(logits).item()
    print(f"Predicted Risk Score: {risk_score:.4f}")

    # --- Step 4: Calibrating Risk to Time-to-Event ---
    print("\n--- Step 4: Calibrating Risk to Time-to-Event ---")
    
    try:
        hazard_ratio = risk_score
        patient_survival_probs = calibration_data['baseline_survival'] ** hazard_ratio
        time_points = calibration_data['time_points']

        time_points_interp = np.flip(time_points)
        patient_survival_probs_interp = np.flip(patient_survival_probs)
        
        min_survival_prob = patient_survival_probs_interp[0]

        if 0.5 > min_survival_prob:
            predicted_time_months = np.interp(0.5, patient_survival_probs_interp, time_points_interp)
            print(f"Predicted Median Time to Recurrence: {predicted_time_months:.2f} months")
        else:
            print("Patient is low risk. Extrapolating time beyond max follow-up.")
            y2, y1 = patient_survival_probs[-1], patient_survival_probs[-2]
            x2, x1 = time_points[-1], time_points[-2]
            
            if (x2 - x1) == 0 or (y2 - y1) == 0:
                 predicted_time_months = x2 * 1.1
            else:
                m = (y2 - y1) / (x2 - x1)
                predicted_time_months = x2 + (0.5 - y2) / m

            if predicted_time_months < x2:
                 predicted_time_months = x2
            print(f"Extrapolated Time to Recurrence: {predicted_time_months:.2f} months")

    except Exception as e:
        print(f"Could not calculate median survival time due to: {e}. Defaulting to -1.")
        predicted_time_months = -1.0
        
    return predicted_time_months

