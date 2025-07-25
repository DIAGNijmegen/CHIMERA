

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
    
    predicted_time_months = -1.0 # Default value
    
    try:
        calibration_type = calibration_data.get('type')
        print(f"Using calibration type: '{calibration_type}'")

        if calibration_type == 'linear_risk_to_time':
            # Formula: time = m * risk_score + b
            slope = calibration_data['slope']
            intercept = calibration_data['intercept']
            
            predicted_time_months = slope * risk_score + intercept
            
            # Ensure time prediction is not negative
            predicted_time_months = max(0, predicted_time_months)
            
            print(f"Predicted Time to Recurrence (from linear fit): {predicted_time_months:.2f} months")

        elif calibration_type == 'exp_risk_to_time':
            # Formula: time = a * exp(-b * risk_score)
            a = calibration_data['a']
            b = calibration_data['b']
            
            predicted_time_months = a * np.exp(-b * risk_score)

            print(f"Predicted Time to Recurrence (from exponential fit): {predicted_time_months:.2f} months")
            
        else:
            print(f"⚠️ Warning: Unknown calibration type '{calibration_type}'. Cannot calculate time.")

    except Exception as e:
        print(f"Could not calculate time due to an error: {e}. Defaulting to -1.")
        predicted_time_months = -1.0
        
    return predicted_time_months

