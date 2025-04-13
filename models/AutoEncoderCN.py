import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    1D сверточный автоэнкодер для временных рядов.
    Вход: (batch_size, channels=51, sequence_length)
    """

    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=51, out_channels=31, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=31, out_channels=19, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=19, out_channels=11, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=11, out_channels=11, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.ConvTranspose1d(in_channels=11, out_channels=19, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=19, out_channels=31, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=31, out_channels=51, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через автоэнкодер.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, channels=51, sequence_length)

        Returns:
            torch.Tensor: Восстановленный тензор такой же формы
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
