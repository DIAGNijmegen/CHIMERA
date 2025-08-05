# Aggregators/inference/prediction.py
import torch

def predict_case(model, pathology_features, clinical_features):
    """
    Predict BRS3 probability and label for a single case.
    """
    with torch.no_grad():
        output_dict = model.forward_no_loss(
            h=pathology_features.unsqueeze(0),  # add batch dim
            clinical_features=clinical_features.unsqueeze(0)
        )
        logits = output_dict["logits"]
        prob_brs3 = torch.sigmoid(logits).item()  # binary classification
        pred_label = int(prob_brs3 > 0.5)

    return prob_brs3, pred_label

