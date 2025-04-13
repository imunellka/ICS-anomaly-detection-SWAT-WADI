import os
from abc import ABC

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader


def impute_batch(x_val, x_id, m_val):

    ax_val = x_val.squeeze(-1)
    ax_id = x_id
    am_val = m_val.squeeze(-1)

    id_unique = torch.unique(ax_id[am_val == 0])

    avg_val = ax_val.sum(dim=1) / (am_val == 0).to(torch.float32).sum(dim=1)
    avg_val = avg_val.unsqueeze(1).repeat(1, ax_val.size()[1])
    ax_val = ax_val * (am_val == 0) + avg_val * (am_val == 1)

    for i in id_unique:
        id_mask = (1 - am_val) * ((ax_id == i)).to(torch.float32)
        id_avg = (ax_val * id_mask).sum(dim=1) / id_mask.sum(dim=1)
        id_avg = id_avg.unsqueeze(1).repeat(1, ax_val.size()[1])

        fill_mask = am_val * ((ax_id == i)).to(torch.float32)
        ax_val = ax_val * (1 - fill_mask) + id_avg * fill_mask

    return ax_val.unsqueeze(-1)


class TokenEncoder(nn.Module):
    """
    Кодирует токен в вектор заданной размерности.

    Каждый токен имеет три поля:
        'timestamp' - время, число;
        'id' - идентификатор, категория, натуральное;
        'values' - значение, число.

    Токены могут содержать пропуски. Пропущенные занчения заполняются 0.

    Args:
        d_model: (int) размер векторного представления токена (required).
        n_ids: (int) число id токенов, включая пропуски (required).
    Examples:
        token_encoder = TokenEncoder(d_model=64, n_ids=3)
    """

    def __init__(self, d_input=40 , d_model=64, use_avg_imputing=False):
        super().__init__()

        self.use_avg_imputing = use_avg_imputing

        self.linear_1 = nn.Linear(d_input*2, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x, m):
        """
        Args:
        :param x_t: (tensor) значения поля 'timestamps' с пропусками. Размер: (B, T, 1)
        :param x_id: (tensor) значения поля 'id' с пропусками. Размер: (B, T)
        :param x_val: (tensor) значения поля 'value' с пропусками. Размер: (B, T, 1)
        :param m_t: (tensor) маска из {0, 1} для 'timestamps'. 1- пропуск, 0 - известное значение. Размер: (B, T, 1)
        :param m_id: (tensor) маска из {0, 1} для 'id'. 1- пропуск, 0 - известное значение. Размер: (B, T, 1)
        :param m_val: (tensor) маска из {0, 1} для 'value'. 1- пропуск, 0 - известное значение. Размер: (B, T, 1)
        :return: (tensor) векторные представления токенов. Размер: (B, T, d_model)
        """

        # if self.use_avg_imputing:
        #     x_val = impute_batch(x_val, x_id, m_val)
        #     x_t = impute_batch(x_t, x_id, m_t)

        xm = torch.cat((x, m), dim=-1)

        xm = F.relu(self.linear_1(xm))
        xm = F.relu(self.output(xm))
        return xm


class TokenDecoder(nn.Module):
    """
        Декодирует токен из вектор заданной размерности.

        Каждый токен имеет три поля:
            'timestamp' - время, число;
            'id' - идентификатор, категория, натуральное;
            'values' - значение, число.

        Args:
            d_model: (int) размер векторного представления токена (required).
            n_ids: (int) число id токенов, включая пропуски (required).
        Examples:
            token_decoder = TokenDecoder(d_model=64, n_ids=3)
        """
    def __init__(self, d_model=64, d_output=40):
        super().__init__()
        self.input = nn.Linear(d_model, d_model)
        self.linear_mu = nn.Linear(d_model, d_output)
        self.linear_logsigma = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = F.relu(self.input(x))
        x_mu = self.linear_mu(x)
        x_logsigma = self.linear_logsigma(x)
        return x_mu, x_logsigma


class Head(nn.Module):
    """ One head of masked self-attention """

    def __init__(self, d_model, head_size, block_size, dropout=0.1, use_diagonal_mask=True):
        super().__init__()

        self.use_diagonal_mask = use_diagonal_mask

        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('eye', torch.eye(block_size, block_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        if self.use_diagonal_mask:
            wei = wei.masked_fill(self.eye[:T, :T] == 1, float(0))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, d_model, num_heads, head_size, block_size, dropout=0.1, use_diagonal_mask=True):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size, block_size,
                                         dropout, use_diagonal_mask) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, d_model, n_head, n_steps, dropout=0.1, use_diagonal_mask=True, use_skip=True):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.use_skip = use_skip
        head_size = d_model // n_head
        self.sa = MultiHeadAttention(d_model, n_head, head_size, n_steps, dropout, use_diagonal_mask)
        self.ffwd = FeedFoward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_skip:
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(self.ln1(x))
            x = self.ffwd(self.ln2(x))
        return x


import math


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_steps=20, n_head=1, n_layer=2, dropout=0.1,
                 use_diagonal_mask=True, use_skip=True, use_pos_encoding=True):
        super().__init__()
        self.use_pos_encoding = use_pos_encoding

        self.l1 = nn.Linear(d_model, d_model)
        self.pe = PositionalEncoding(d_model, dropout, n_steps)
        self.blocks = nn.Sequential(*[Block(d_model=d_model, n_head=n_head, n_steps=n_steps, dropout=dropout,
                                            use_diagonal_mask=use_diagonal_mask, use_skip=use_skip) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.use_pos_encoding:
            x = self.pe(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return x


class MaskedTranEncoder(ABC):

    def __init__(self, n_steps=20, d_input=2, d_model=64, n_head=1, n_layer=2, batch_size=32,
                       n_epochs=10, lr=0.001, weight_decay=0, dropout=0.1, use_diagonal_mask=True, use_skip=True,
                       use_avg_imputing=False, use_pos_encoding=True, device='cpu'):

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.use_avg_imputing = use_avg_imputing
        self.use_pos_encoding = use_pos_encoding
        self.device = device

        self.token_encoder = TokenEncoder(d_input=d_input, d_model=d_model, use_avg_imputing=False).to(self.device)
        self.token_decoder = TokenDecoder(d_model=d_model, d_output=d_input).to(self.device)
        self.model = TransformerEncoder(d_model=d_model, n_steps=n_steps, n_head=n_head, n_layer=n_layer,
                                        dropout=dropout, use_diagonal_mask=use_diagonal_mask,
                                        use_skip=use_skip, use_pos_encoding=self.use_pos_encoding).to(self.device)

        self.opt = torch.optim.Adam(list(self.token_encoder.parameters()) + \
                                    list(self.token_decoder.parameters()) + \
                                    list(self.model.parameters()),
                                    lr=lr, weight_decay=weight_decay)

        self.loss_history = []

    def gaussian_nll_loss(self, target, mu, sigma, mask=None, eps=10 ** -6):
        sigma = sigma + eps
        loss = -1. * torch.log(sigma) - (target - mu) ** 2 / (2 * sigma ** 2)
        if mask is not None:
            n_ones = mask[mask == 1].numel()
            if n_ones == 0:
                loss = float(0)
            else:
                loss = loss * mask
                loss = -loss.sum() / n_ones
        else:
            loss = -loss.mean()
        return loss

    def masked_ce_loss(self, loss_vals, mask=None):
        if mask is not None:
            n_ones = mask[mask == 1].numel()
            if n_ones == 0:
                loss = float(0)
            else:
                loss_vals = loss_vals * mask
                loss = loss_vals.sum() / n_ones
        else:
            loss = loss_vals.mean()
        return loss

    def fit(self, X, mask_missing, mask_art_missing):

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask_missing = torch.tensor(mask_missing, dtype=torch.float32, device=self.device)
        mask_art_missing = torch.tensor(mask_art_missing, dtype=torch.float32, device=self.device)

        mask_missing[mask_art_missing == 1] = float(1)

        X_missing = torch.tensor(X, dtype=torch.float32, device=self.device)
        X_missing[mask_missing == 1] = float(0)

        train_dataset = TensorDataset(X, X_missing, mask_missing, mask_art_missing)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        loss_ce = nn.CrossEntropyLoss(reduction='none')

        self.token_encoder.train()
        self.token_decoder.train()
        self.model.train()

        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch} starting...")
            for batch in train_dataloader:

                x_batch, x_m_batch, m_batch, m_art_batch = batch

                token_embed = self.token_encoder(x_m_batch, m_batch)
                output_embed = self.model(token_embed)
                x_pred_mu, x_pred_logsigma = self.token_decoder(output_embed)

                loss_val = self.gaussian_nll_loss(x_pred_mu, x_batch, torch.exp(x_pred_logsigma), 1 - m_batch) + \
                           self.gaussian_nll_loss(x_pred_mu, x_batch, torch.exp(x_pred_logsigma), m_art_batch)
                loss = loss_val

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.loss_history.append(loss.detach().numpy())

        # self.token_encoder.eval()
        # self.token_decoder.eval()
        # self.model.eval()

    def sample_cat_values(self, probas):
        # sam = [torch.multinomial(p, num_samples=1).unsqueeze(0) for p in probas]
        # sam = torch.cat(sam, dim=0)
        sam = torch.distributions.categorical.Categorical(probas).sample().unsqueeze(2)
        return sam

    def _predict(self, X, mask_missing):

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        mask_missing = torch.tensor(mask_missing, dtype=torch.float32, device=self.device)

        with torch.no_grad():

            self.token_encoder.eval()
            self.token_decoder.eval()
            self.model.eval()

            token_embed = self.token_encoder(X, mask_missing)
            output_embed = self.model(token_embed)
            X_pred_mu, X_pred_logsigma = self.token_decoder(output_embed)

        X_pred_sam = torch.normal(X_pred_mu, torch.exp(X_pred_logsigma))

        return X_pred_sam, X_pred_mu, X_pred_logsigma

    def predict(self, X, mask_missing):
        X_pred_sam, X_pred_mu, X_pred_logsigma = self._predict(X, mask_missing)
        return X_pred_sam.detach().numpy(), X_pred_mu.detach().numpy(), X_pred_logsigma.detach().numpy()

    def anomaly_scores(self, X, missing_proba=0.5):

        mask_missing = 1 * (np.random.random(X.shape) <= missing_proba)
        X_missing = X.copy()
        X_missing[mask_missing == 1] = 0

        X_pred_sam, X_pred_mu, X_pred_logsigma = self._predict(X_missing, mask_missing)

        norm = torch.distributions.normal.Normal(X_pred_mu, torch.exp(X_pred_logsigma))
        scores = -norm.log_prob(torch.tensor(X)).mean(-1)

        return scores.detach().numpy()

    def average_anomaly_score(self, X, missing_proba=0.5):

        scores = self.anomaly_scores(X, missing_proba=0.5)

        return scores.mean(-1)

    # def predict(self, tokens, mask_missing, impute_known=False):
    #
    #     tokens = torch.tensor(tokens, dtype=torch.float32, device=self.device)
    #     mask_missing = torch.tensor(mask_missing, dtype=torch.float32, device=self.device)
    #
    #     tokens_imputed = []
    #     with torch.no_grad():
    #         for i in np.arange(0, len(tokens)):
    #             seq_imputed = self._one_seq_prediction(tokens[[i]], mask_missing[[i]])
    #             if not impute_known:
    #                 seq_imputed = seq_imputed * mask_missing[[i]] + \
    #                               tokens[[i]] * (1 - mask_missing[[i]])
    #             tokens_imputed.append(seq_imputed)
    #     tokens_imputed = torch.cat(tokens_imputed, dim=0)
    #
    #     return tokens_imputed