import torch
import torch.nn as nn

class ABMIL_Fusion_RNA_Clinical(nn.Module):
    def __init__(self, in_dim=1024, rna_dim=None, clinical_dim=None, risk_output_dim=1):
        super().__init__()

        assert rna_dim is not None, "rna_dim must be provided"
        assert clinical_dim is not None, "clinical_dim must be provided"

        # WSI feature embedding + attention pooling
        self.embedding = nn.Linear(in_dim, 512)
        self.attention = nn.Linear(512, 1)

        # RNA projection network
        self.rna_net = nn.Sequential(
            nn.Linear(rna_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

        # Clinical projection network
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )

        # Final risk prediction head
        self.risk_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 256 + 64, risk_output_dim)
        )

    def forward(self, x_bag, x_rna, x_clinical, return_l1=False):
        # === Pathology attention pooling ===
        h = torch.tanh(self.embedding(x_bag))             # [B, N, 512]
        a = torch.softmax(self.attention(h), dim=1)       # [B, N, 1]
        z_path = torch.sum(a * h, dim=1)                  # [B, 512]

        # === RNA projection ===
        z_rna = self.rna_net(x_rna)                       # [B, 256]

        # === Clinical projection ===
        z_clinical = self.clinical_net(x_clinical)        # [B, 64]

        # === Feature fusion ===
        z = torch.cat([z_path, z_rna, z_clinical], dim=-1)  # [B, 832]

        # === Risk prediction ===
        risk = self.risk_head(z).squeeze(1)               # [B]

        # === Optional L1 penalty on RNA weights ===
        if return_l1:
            l1_penalty = sum(torch.norm(p, p=1) for p in self.rna_net.parameters())
            return {'risk': risk}, l1_penalty
        else:
            return {'risk': risk}
