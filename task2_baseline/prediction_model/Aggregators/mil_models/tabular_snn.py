import torch.nn as nn

class TabularSNN(nn.Module):
    def __init__(self, clinical_in_dim, dropout_p=0.3):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(clinical_in_dim, 64),
            nn.LayerNorm(64),   
            nn.SELU(),
            nn.AlphaDropout(dropout_p),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.AlphaDropout(dropout_p),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.SELU(),
            nn.AlphaDropout(dropout_p),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.SELU(),
            nn.AlphaDropout(dropout_p)
        )

    def forward(self, x):
        return self.mlp(x)
