import torch.nn as nn
from mil_models.model_abmil_fusion import ABMIL_Fusion_RNA_Clinical
from mil_models.model_survival import CoxLossModel


def create_downstream_model(args, mode='survival'):
    assert args.model_type == 'ABMIL_Fusion_RNA_Clinical', "Only ABMIL_Fusion_RNA_Clinical is supported in this setup."

    model = ABMIL_Fusion_RNA_Clinical(
        in_dim=args.in_dim,
        rna_dim=args.rna_dim,
        clinical_dim=args.clinical_dim,
        risk_output_dim=1
    )

    if mode == 'survival':
        model = CoxLossModel(model)

    return model
