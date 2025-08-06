# Aggregators/inference/prediction.py
import torch

def predict_case(model, pathology_features, clinical_features):
    """
    Predict BRS3 probability and label for a single case (Task 2 classification).
    
    Args:
        model: Trained ABMIL_Fusion model.
        pathology_features: Tensor of extracted pathology features (num_patches x feature_dim).
        clinical_features: Tensor of processed clinical features (num_clinical_features).
    
    Returns:
        prob_brs3 (float): Probability of BRS3 (positive class).
        pred_label (int): Predicted label (1 = BRS3, 0 = BRS1/2).
    """
    model.eval()

    # Add batch dimension
    pathology_features = pathology_features.unsqueeze(0)  # (1, num_patches, feature_dim)
    clinical_features = clinical_features.unsqueeze(0)    # (1, num_clinical_features)

    print("\n--- Step 3: Running Model Inference ---")
    with torch.no_grad():
        output_dict = model.forward(pathology_features, clinical_features)
        logits = output_dict["logits"]

    # Binary classification: Apply sigmoid to get probability of BRS3
    prob_brs3 = torch.sigmoid(logits).item()
    pred_label = int(prob_brs3 > 0.5)

    print(f"Predicted BRS3 Probability: {prob_brs3:.4f}")
    print(f"Predicted Label: {'BRS3' if pred_label == 1 else 'BRS1/2'}")

    return prob_brs3, pred_label
