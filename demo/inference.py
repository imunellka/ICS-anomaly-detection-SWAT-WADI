import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from data.WaterSystemDataset import WaterSystemDataset
from models.AutoEncoderCN import Autoencoder
from models.MaskedTranEncoder import MaskedTranEncoder
from models.ProbabilisticTranEncoder import ProbTransAE
from models.TransformerbasedEncoder import TransformerbasedEncoder
from models.WaterSystemAnomalyTrasformer import WaterSystemAnomalyTransformer
import json
import torch.nn.functional as F

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
    model = Autoencoder()
    print(model_path)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    return model

model_to_pred_file = {
    "IsolationForest.pkl": "demo/preds/preds_if.npy",
    "AutoEncoderCN.pth": "demo/preds/preds_AutoEncoder_CN.npy",
    "TransformerEncoder.pth": "demo/preds/preds_TranEncoder.npy",
    "ProbabilisticTranEncoder.pth": "demo/preds/preds_ProbabilisticTranEncoder.npy",
    "WaterSystemBertImputer.pth": "demo/preds/preds_WaterSystemBertImputer.npy",
    "WaterSystemBertImputerv2.pth": "demo/preds/preds_WaterSystemBertImputer_v2.npy",
    "WaterSystemBertImputerv3.pth": "demo/preds/preds_WaterSystemBertImputer_v3.npy",
    "WaterSystemBertImputerv4.pth": "demo/preds/preds_WaterSystemBertImputer_v4.npy",
}

def predict_anomalies(model_path, model_name="AutoEncoderCN.pth", use_saved_preds=False):
    device = "cpu"

    test_dataset = WaterSystemDataset(
        "demo/test_swat_cropped.npy",
        feature_idx=list(range(51)),
        start_idx=0,
        end_idx=100_000,
        window_size=30,
        sliding=1
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"🧠 Инференс модели: {model_name}")

    if use_saved_preds and model_name in model_to_pred_file:
        y_attack_scores = np.load(model_to_pred_file[model_name])
        y_pred = y_attack_scores
    else:

        model = load_model(model_path, device)
        scores = []
        with torch.no_grad():
            for x in test_loader:
                x_data = x[0].to(device)
                y_hat = model(x_data.permute(0, 2, 1))
                batch_scores = F.mse_loss(y_hat, x_data.permute(0, 2, 1), reduction="none")
                scores.extend(batch_scores.mean(1))

        y_attack_scores_w = np.array(scores)

        test_np = np.load("demo/test_swat_cropped.npy")
        y_attack_scores = transform_anomaly_scores(test_np, y_attack_scores_w)

        threshold =  700
        y_pred = (y_attack_scores > threshold).astype(int)

    return y_pred, y_attack_scores
