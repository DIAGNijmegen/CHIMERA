import torch.nn as nn

class TabularSNN(nn.Module):
    def __init__(self, clinical_in_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(clinical_in_dim, 64),
            nn.SELU(),
            nn.AlphaDropout(0.1),

            nn.Linear(64, 128),
            nn.SELU(),
            nn.AlphaDropout(0.1),

            nn.Linear(128, 256),
            nn.SELU(),
            nn.AlphaDropout(0.1),

            nn.Linear(256, 512),
            nn.SELU()
        )

    def forward(self, x):
        return self.mlp(x)
