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
    loaded_imputer = WaterSystemAnomalyTransformer.load(model_path)
    loaded_imputer.to(device)
    loaded_imputer.eval()

    return loaded_imputer

def get_windows(X, window_size=40, batch_size=10000, stride=1):
    result = []
    for i in range(0, len(X) - window_size + 1, batch_size):
        end = min(i + batch_size, len(X) - window_size + 1)
        batch = [X[j:j + window_size] for j in range(i, end, stride)]
        result.extend(batch)
    return np.array(result)

def predict_anomalies(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset_np = np.load("demo/test_swat_cropped.npy", allow_pickle=True).take(list(range(51)), axis=1)
    test_dataset_w = get_windows(test_dataset_np,  window_size=30, stride = 1)

    model = load_model(model_path, device)

    y_attack_scores_w = model.anomaly_scores(test_dataset_w, 0.1, 32)
    y_attack_scores = transform_anomaly_scores(test_dataset_np, y_attack_scores_w)

    threshold =  865.8972736409505

    y_pred = (y_attack_scores > threshold).astype(int)

    return y_pred, y_attack_scores
