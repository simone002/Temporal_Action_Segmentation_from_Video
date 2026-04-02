import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM per catturare dipendenze temporali.
    Input:  (B, T, feat_dim)
    Output: (B, T, num_classes)
    """

    def __init__(
        self,
        feat_dim: int    = 1024,
        num_classes: int = 106,
        hidden: int      = 512,
        n_layers: int    = 2,
        dropout: float   = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = feat_dim,
            hidden_size   = hidden,
            num_layers    = n_layers,
            batch_first   = True,
            dropout       = dropout if n_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )
        lstm_out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, feat_dim)
        out, _ = self.lstm(x)       # (B, T, hidden*2)
        return self.head(out)       # (B, T, num_classes)