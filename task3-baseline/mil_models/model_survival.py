import torch
import torch.nn as nn

class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, risk_pred, survival_time, event):
        # Sort by descending survival time
        sorted_idx = torch.argsort(survival_time, descending=True)
        risk_pred = risk_pred[sorted_idx]
        event = event[sorted_idx]

        hazard_ratio = torch.exp(risk_pred)
        log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_pred - log_cum_hazard
        censored_likelihood = uncensored_likelihood * event
        loss = -torch.mean(censored_likelihood)
        return loss

class CoxLossModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = CoxLoss()

    def forward(self, x, rna=None, clinical=None, survival_time=None, censorship=None, return_l1=False):
        # Forward through base model
        if return_l1:
            out, l1_penalty = self.base_model(x, rna, clinical, return_l1=True)
        else:
            out = self.base_model(x, rna, clinical)
            l1_penalty = None

        risk = out['risk']

        # Compute loss only if survival labels are provided
        if survival_time is not None and censorship is not None:
            loss = self.loss_fn(risk, survival_time, censorship)
            if return_l1:
                return out, loss, l1_penalty
            return out, loss

        return out
