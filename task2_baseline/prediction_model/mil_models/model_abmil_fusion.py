import torch
import torch.nn as nn
from mil_models.tabular_snn import TabularSNN


class ABMIL_Fusion(nn.Module):
    def __init__(self, in_dim, clinical_in_dim, n_classes, gate=True, dropout_p=0.5):
        super().__init__()
        self.gate = gate

        # === MIL embedding branch with configurable dropout ===
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p)  # configurable dropout
        )

        # Attention branch
        self.attention = nn.Linear(512, 1)

        # === Clinical branch with LayerNorm (safe for batch size = 1) ===
        self.tabular_net = TabularSNN(clinical_in_dim=clinical_in_dim, dropout_p=0.3)

        # === Safe dummy forward to get output dimension ===
        with torch.no_grad():
            self.tabular_net.eval()  # prevent any norm layers from training stats
            dummy_input = torch.zeros(1, clinical_in_dim)
            clinical_out_dim = self.tabular_net(dummy_input).shape[1]
            self.tabular_net.train()

        # === Fusion classifier ===
        self.classifier = nn.Linear(512 + clinical_out_dim, n_classes)

    def forward(self, x_bag, x_clinical):
        # MIL embedding
        h = self.embedding(x_bag)
        a = torch.softmax(self.attention(h), dim=1)
        z = torch.sum(a * h, dim=1)

        # Clinical embedding
        z_tab = self.tabular_net(x_clinical)

        # Optional gating
        if self.gate:
            z_tab = 0.5 * z_tab  # less aggressive scaling than 0.2

        # Fusion
        z_fusion = torch.cat((z, z_tab), dim=-1)
        logits = self.classifier(z_fusion)

        return {'logits': logits, 'loss': None, 'attention': a}

    def attention_entropy_loss(self, attention_weights):
        """Optional regularization to avoid overconfident attention."""
        entropy = -torch.mean(
            torch.sum(attention_weights * torch.log(attention_weights + 1e-6), dim=1)
        )
        return entropy
