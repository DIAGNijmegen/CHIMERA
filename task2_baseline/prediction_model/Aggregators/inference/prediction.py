# Aggregators/inference/prediction.py

import torch


def predict_case(model, pathology_features, clinical_features):
    """
    Predict BRS3 probability and label for a single case (Task 2 classification).

    Args:
        model: Trained ABMIL_Fusion model.
        pathology_features (Tensor): (num_patches, feat_dim)
        clinical_features (Tensor): (num_clinical_features,)

    Returns:
        prob_brs3 (float): Predicted probability of BRS3.
        pred_label (int): Predicted label (1 for BRS3, 0 for BRS1/2).
    """
    model.eval()

    # Ensure inputs are batched
    if pathology_features.ndim == 2:
        pathology_features = pathology_features.unsqueeze(0)  # (1, num_patches, feat_dim)
    if clinical_features.ndim == 1:
        clinical_features = clinical_features.unsqueeze(0)    # (1, num_clinical_features)

    print("\n--- Step 3: Running Model Inference ---")
    with torch.no_grad():
        output_dict = model.forward(pathology_features, clinical_features)
        logits = output_dict["logits"]  # shape: (1,)

    # Binary classification: Apply sigmoid
    prob_brs3 = torch.sigmoid(logits).item()
    pred_label = int(prob_brs3 > 0.5)

    print(f"[RESULT] Predicted BRS3 Probability: {prob_brs3:.4f}")
    print(f"[RESULT] Predicted Label: {'BRS3' if pred_label == 1 else 'BRS1/2'}")

    return prob_brs3, pred_label
