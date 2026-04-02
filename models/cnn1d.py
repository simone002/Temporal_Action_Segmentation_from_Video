import torch.nn as nn


class CNN1DModel(nn.Module):
    """
    Baseline: CNN 1D lungo la dimensione temporale.
    Input:  (B, T, feat_dim)
    Output: (B, T, num_classes)
    """

    def __init__(
        self,
        feat_dim: int    = 1024,
        num_classes: int = 106,
        hidden: int      = 512,
        n_layers: int    = 4,
        kernel_size: int = 3,
        dropout: float   = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters = lambda: None  # placeholder

        layers = []
        in_ch = feat_dim
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, hidden, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_ch = hidden

        self.encoder = nn.Sequential(*layers)
        self.head    = nn.Conv1d(hidden, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, T, feat_dim) → (B, feat_dim, T)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.head(x)
        return x.permute(0, 2, 1)  # (B, T, num_classes)