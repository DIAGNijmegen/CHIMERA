import torch
import torch.nn as nn
from prediction_model.Aggregators.inference.tabular_snn import TabularSNN  # see step 2

class ABMIL_Fusion(nn.Module):
    def __init__(self, in_dim, clinical_in_dim, n_classes, gate=True, dropout_p=0.5):
        super().__init__()
        self.gate = gate
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout_p)
        )
        self.attention = nn.Linear(512, 1)
        self.tabular_net = TabularSNN(clinical_in_dim=clinical_in_dim, dropout_p=0.3)
        self.classifier = nn.Linear(512 + 512, n_classes)  # TabularSNN outputs 512
    def forward(self, x_bag, x_clinical):
        h = self.embedding(x_bag)                 # [B?, N, 512]
        a = torch.softmax(self.attention(h),  dim=1)
        z = torch.sum(a * h, dim=1)               # [B?, 512]
        z_tab = self.tabular_net(x_clinical)      # [B?, 512]
        if self.gate: z_tab = 0.5 * z_tab
        logits = self.classifier(torch.cat([z, z_tab], dim=-1))
        return {"logits": logits, "attention": a}
