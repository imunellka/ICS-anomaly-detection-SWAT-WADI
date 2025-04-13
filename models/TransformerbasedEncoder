import math
import torch
import torch.nn as nn


class TransformerbasedEncoder(nn.Module):
    """
    Автоэнкодер на основе Transformer для временных рядов.

    Args:
        input_dim (int): Размерность входных признаков (каналов).
        embed_dim (int): Размерность эмбеддингов.
        n_heads (int): Кол-во голов в Multi-head attention.
        ff_dim (int): Размерность feedforward слоя внутри Transformer.
        num_layers (int): Количество энкодерных слоев.
        dropout (float): Дропаут.
        max_seq_len (int): Максимальная длина входной последовательности.
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
        self.positional_encoding = self.create_positional_encoding(embed_dim, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # позволяет не транспонировать x
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 46),
            nn.ReLU(),
            nn.Linear(46, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через модель.

        Args:
            x (torch.Tensor): Тензор формы (batch_size, input_dim, seq_len)

        Returns:
            torch.Tensor: Реконструированный выход той же формы
        """
        batch_size, input_dim, seq_len = x.size()
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.input_projection(x) + self.positional_encoding[:seq_len, :].to(x.device)
        encoded = self.encoder(x)  # (B, T, embed_dim)
        output = self.regressor(encoded)  # (B, T, input_dim)
        return output.permute(0, 2, 1)  # (B, input_dim, T)

    @staticmethod
    def create_positional_encoding(embed_dim: int, max_seq_len: int) -> torch.Tensor:
        """
        Создает синусоиду позиционного кодирования.

        Args:
            embed_dim (int): Размерность эмбеддинга.
            max_seq_len (int): Максимальная длина последовательности.

        Returns:
            torch.Tensor: Позиционное кодирование размера (max_seq_len, embed_dim)
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
