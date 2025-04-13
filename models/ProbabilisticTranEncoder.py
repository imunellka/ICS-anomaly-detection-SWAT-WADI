import math
import torch
import torch.nn as nn
from torch import Tensor


class ProbTransAE(nn.Module):
    """
    Probabilistic Transformer-based Autoencoder for multivariate time series anomaly detection.
    Outputs per-timepoint distributions (mean and variance) instead of a deterministic reconstruction.

    Args:
        input_dim (int): Number of input features (channels).
        embed_dim (int): Embedding dimension for transformer.
        n_heads (int): Number of attention heads.
        ff_dim (int): Feedforward layer dimension in the transformer.
        num_layers (int): Number of transformer layers.
        dropout (float): Dropout rate in transformer.
        max_seq_len (int): Maximum sequence length for positional encoding.
    """
    def __init__(
        self,
        input_dim: int = 51,
        embed_dim: int = 36,
        n_heads: int = 2,
        ff_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 30,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = self._create_positional_encoding(embed_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mu_regressor = nn.Sequential(
            nn.Linear(embed_dim, 46),
            nn.ReLU(),
            nn.Linear(46, input_dim)
        )

        self.sigma_regressor = nn.Sequential(
            nn.Linear(embed_dim, 46),
            nn.ReLU(),
            nn.Linear(46, input_dim)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim, seq_len)

        Returns:
            mu (Tensor): Mean tensor of shape (batch_size, input_dim, seq_len)
            sigma (Tensor): Std dev tensor of shape (batch_size, input_dim, seq_len)
        """
        batch_size, input_dim, seq_len = x.size()
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)
        x = self.input_projection(x) + self.positional_encoding[:seq_len, :].to(x.device)
        encoded = self.encoder(x)

        mu = self.mu_regressor(encoded)
        sigma = self.sigma_regressor(encoded)

        return mu.permute(0, 2, 1), sigma.permute(0, 2, 1)

    @staticmethod
    def _create_positional_encoding(embed_dim: int, max_seq_len: int) -> Tensor:
        """
        Generate sinusoidal positional encodings.

        Returns:
            Tensor: Positional encoding tensor of shape (max_seq_len, embed_dim)
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
