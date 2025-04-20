import numpy as np
import torch
from torch.utils.data import DataLoader
from data.WaterSystemDataset import WaterSystemDataset
from models.AutoEncoderCN import AutoEncoderCN
from models.MaskedTranEncoder import MaskedTranEncoder
from models.ProbabilisticTranEncoder import ProbabilisticTranEncoder
from models.TransformerbasedEncoder import TransformerbasedEncoder
from models.WaterSystemAnomalyTrasformer import WaterSystemAnomalyTrasformer
import json
import torch.nn.functional as F
import os

def transform_anomaly_scores(X_attack, scores):
    y_pred = np.zeros(X_attack.shape[0])
    counts = np.zeros(X_attack.shape[0])
    w_size = scores.shape[1]
    for i in range(len(scores)):
        i_score = scores[i]
        y_pred[i:i+w_size] += i_score
        counts[i:i+w_size] += 1
    return y_pred / counts

def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint["config"]
    model = AutoEncoderCN(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, config


def predict_anomalies(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = WaterSystemDataset(
        "demo/test_swat_cropped.npy",
        feature_idx=list(range(51)),
        start_idx=0,
        end_idx=100_000,
        window_size=30,
        sliding=1
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model, config = load_model(model_path, device)

    scores = []
    with torch.no_grad():
        for x in test_loader:
            x_data = x[0].to(device)
            y_hat = model(x_data)
            batch_scores = F.mse_loss(reconstructed, X_data, reduction="none")
            scores.extend(batch_scores)

    y_attack_scores_w = np.array(scores)

    test_np = np.load("demo/test_swat_cropped.npy")
    y_attack_scores = transform_anomaly_scores(test_np, y_attack_scores_w)

    y_true = test_np[:, -1].astype(int)

    threshold =  70000

    y_pred = (y_attack_scores > threshold).astype(int)

    return y_pred, y_attack_scores
