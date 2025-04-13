import torch
import torch.nn.functional as F
import numpy as np
from tqdm.notebook import tqdm
from torch import nn
from torch.utils.data import DataLoader
from typing import Union


def train_probabilistic_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int = 40,
    device: Union[str, torch.device] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    use_wandb: bool = False,
) -> list[float]:
    """
    Train a probabilistic transformer autoencoder using negative log-likelihood loss.

    Args:
        model (nn.Module): Transformer model outputting (mu, sigma)
        train_loader (DataLoader): PyTorch DataLoader with training data
        optimizer (Optimizer): Optimizer for training
        n_epochs (int): Number of training epochs
        device (str | torch.device): Device for training
        use_wandb (bool): Whether to use Weights & Biases for logging

    Returns:
        list[float]: Training loss history
    """
    model.to(device)
    train_losses = []

    if use_wandb:
        import wandb
        wandb.init(project="experiment_1_transformer_autoencoder", name="probabilistic_run")

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.train()
        epoch_losses = []

        for X_batch in train_loader:
            X = X_batch[0].permute(0, 2, 1).to(device)
            optimizer.zero_grad()

            mu, sigma = model(X)

            sigma = torch.exp(sigma)

            dist = torch.distributions.Normal(mu, sigma)
            nll = -dist.log_prob(X).mean()

            nll.backward()
            optimizer.step()
            epoch_losses.append(nll.item())

        epoch_loss = np.mean(epoch_losses)
        train_losses.append(epoch_loss)

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss})

        print(f"Epoch [{epoch + 1}/{n_epochs}] - Train Loss: {epoch_loss:.6f}")

    if use_wandb:
        wandb.finish()

    return train_losses
