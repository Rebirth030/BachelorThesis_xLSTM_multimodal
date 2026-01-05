"""LSTM model definition."""

import torch.nn as nn

class LSTM(nn.Module):
    """Stacked LSTM with a linear output head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        horizon: int,
        dropout: float,
    ):
        """Initialize the LSTM and output head.

        Args:
            n_features: Number of input features per time step.
            hidden_size: Hidden size of the LSTM layers.
            num_layers: Number of stacked LSTM layers.
            horizon: Prediction horizon length.
            dropout: Dropout rate applied between LSTM layers.
        """
        super().__init__()
        assert num_layers >= 1

        self.n_features = n_features
        self.horizon = horizon
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        """Run a forward pass through the LSTM and head.

        Args:
            x: Input tensor shaped [B, S, F].

        Returns:
            Tensor of predictions shaped [B, H].
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
