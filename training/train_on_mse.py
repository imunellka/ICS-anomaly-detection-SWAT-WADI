import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def train_autoencoder(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 50,
    lr: float = 1e-4,
    device: Optional[str] = None,
    use_wandb: bool = True,
    run_name: str = "autoencoder_run",
    project_name: str = "transformer_autoencoder"
) -> Tuple[list, list]:
    """
    Обучает автоэнкодер и логгирует метрики.

    Args:
        model (torch.nn.Module): Модель автоэнкодера.
        train_loader (DataLoader): Даталоадер для обучения.
        val_loader (DataLoader, optional): Даталоадер для валидации.
        n_epochs (int): Количество эпох.
        lr (float): Learning rate.
        device (str, optional): CUDA / CPU.
        use_wandb (bool): Включить wandb логгинг.
        run_name (str): Имя запуска в wandb.
        project_name (str): Имя проекта в wandb.

    Returns:
        Tuple[List[float], List[float]]: Списки потерь train/val по эпохам.
    """

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = F.mse_loss

    if use_wandb:
        wandb.init(project=project_name, name=run_name)
        wandb.watch(model)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        epoch_train_losses = []

        for X_batch in train_loader:
            x = X_batch[0].permute(0, 2, 1).to(device)  # (B, C, T)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = loss_fn(reconstructed, x)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        train_loss = np.mean(epoch_train_losses)
        train_losses.append(train_loss)

        val_loss = None
        if val_loader:
            model.eval()
            epoch_val_losses = []
            with torch.no_grad():
                for X_batch in val_loader:
                    x = X_batch[0].permute(0, 2, 1).to(device)
                    reconstructed = model(x)
                    loss = loss_fn(reconstructed, x)
                    epoch_val_losses.append(loss.item())
            val_loss = np.mean(epoch_val_losses)
            val_losses.append(val_loss)

        if use_wandb:
            log_data = {"epoch": epoch + 1, "train_loss": train_loss}
            if val_loss is not None:
                log_data["val_loss"] = val_loss
            wandb.log(log_data)

        print(f"📦 Epoch [{epoch+1}/{n_epochs}] | Train Loss: {train_loss:.6f}" +
              (f" | Val Loss: {val_loss:.6f}" if val_loss is not None else ""))

    torch.save(model.state_dict(), f"{run_name}_weights.pth")
    print(f"✅ Model saved as '{run_name}_weights.pth'")

    if use_wandb:
        wandb.finish()

    return train_losses, val_losses
