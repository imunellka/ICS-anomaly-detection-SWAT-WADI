import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple


def compute_reconstruction_errors(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет ошибки реконструкции для батчей и сохраняет реальные метки.

    Args:
        dataloader (DataLoader): DataLoader с входами и метками (X, y).
        model (torch.nn.Module): Обученная модель автоэнкодера.
        device (str): Устройство ('cuda' или 'cpu').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - errors: массив среднеквадратичных ошибок реконструкции для каждого примера.
            - labels: соответствующие реальные метки (0 — норма, 1 — аномалия).
    """
    model.eval()
    errors = []
    labels = []

    with torch.no_grad():
        for X_batch in dataloader:
            X_data, y_true = X_batch[0].to(device), X_batch[1].cpu().numpy()
            X_data = X_data.permute(0, 2, 1)  # (B, C, T)

            reconstructed = model(X_data)
            # Покомпонентная MSE ошибка без усреднения
            loss = F.mse_loss(reconstructed, X_data, reduction="none")
            loss = loss.mean(dim=[1, 2])  # Усредняем по всем признакам и длине

            errors.extend(loss.cpu().numpy())
            labels.extend(y_true)

    return np.array(errors), np.array(labels)
