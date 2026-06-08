"""
Architecture (TabTransformer pattern):
  Each feature is treated as a token in the sequence.
  Multi-head self-attention lets features attend to each other.
  This captures interactions between features that tree models miss.

Flow:
  Input (batch, n_features)
    → Feature Embedding   (batch, n_features, d_model)
    → Positional Encoding (batch, n_features, d_model)
    → Transformer Encoder (batch, n_features, d_model)  x n_layers
    → Global Avg Pool     (batch, d_model)
    → Output Head         (batch, 1)
    → squeeze             (batch,)
 """

import logging
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

import config

logger = logging.getLogger(__name__)

# #Sinusoidal Positional Embedding
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1).float()  # from array it become array of arrays
#         div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         # div
#         # [1,
#         #  0.1,
#         #  0.01,
#         #  0.001]
#
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#         # adding a batch
#         # (1, max_len, d_model)


class TransformerEncoderNet(nn.Module):
    def __init__(self, n_features: int,
                 d_model: int = 64,
                 n_head: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 256,
                 dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features

        # feature embedding
        # each scalar feature → d_model dimensions
        # (batch, n_features, 1)
        # og_ret_1d = 0.0023. This layer projects that single number into 64 dimensions so the transformer can work with it.
        self.feature_embeddings  = nn.Linear(1, d_model)
        # (batch, n_features, d_model)

        # learned positional embedding
        self.pos_embedding = nn.Embedding(n_features, d_model)
        self.posdropout=nn.Dropout(p=dropout)
        #Position 0 gets vector [w₀, w₁, ..., w₆₃], position 1 gets a different vector, and so on. These vectors are trainable parameters updated by backpropagation.

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

#If weights start too large, activations explode and gradients become NaN on step 1. If too small, gradients vanish and the model never learns. Xavier is mathematically derived to keep signal magnitude stable across any number of layers."""
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.feature_embeddings.weight)
        nn.init.zeros_(self.feature_embeddings.bias)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        for module in self.output_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x.unsqueeze(-1)                          # (batch, n_features, 1)
        x = self.feature_embeddings(x)               # (batch, n_features, d_model)
        #unsqueeze(0) is often used to add a batch dimension.unsqueeze(1) is often used when creating attention masks.unsqueeze(-1) is commonly used to make features compatible with linear layers, RNNs, GRUs, LSTMs, or broadcasting operations.

        ## (batch, n_features, d_model)
        positions = torch.arange(self.n_features, device=x.device)
        pos_emb = self.pos_embedding(positions)      # (n_features, d_model)
        x = x + pos_emb.unsqueeze(0)
        x = self.posdropout(x)                       # (batch, n_features, d_model)

        #  transformer encoder
        x = self.transformer(x)                      # (batch, n_features, d_model)

        ###— global average pooling
        x = x.mean(dim=1)                            # (batch, d_model)
        # step 5 — output head
        x = self.output_head(x)                      # (batch, 1)
        return x.squeeze(-1)                         # (batch,)



   #TransformerEncoderNet is a pure PyTorch model — it knows nothing about DataFrames, sklearn, or your pipeline. TransformerModel wraps it to look like LGBMRegressor or StackingModel.


class TransformerModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_fitted = False
        self.net_ = None
        self.n_features = None
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.debug("device: %s", self.device)

    def _validate_fitted(self):
        if not self.is_fitted:
            raise ValueError(
                "not fitted — call fit() before predict()"
            )

    def _to_tensor(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return torch.tensor(x, dtype=torch.float32).to(self.device)

    def fit(self, X, y):
        if len(y) < config.MIN_TRAIN_SAMPLES:
            raise ValueError(f"only {len(y)} samples")
        self.n_features_ = X.shape[1]

        self.net_ = TransformerEncoderNet(
            n_features = self.n_features_,
            d_model    = config.TRANSFORMER_PARAMS["d_model"],
            n_head     = config.TRANSFORMER_PARAMS["n_heads"],
            n_layers   = config.TRANSFORMER_PARAMS["n_layers"],
            d_ff       = config.TRANSFORMER_PARAMS["d_ff"],
            dropout    = config.TRANSFORMER_PARAMS["dropout"],
        ).to(self.device)

        X_tensor = self._to_tensor(X)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=config.TRANSFORMER_PARAMS["batch_size"],
                shuffle=False)
        optimizer = torch.optim.AdamW(self.net_.parameters(),
                lr=config.TRANSFORMER_PARAMS["learning_rate"],
                weight_decay=config.TRANSFORMER_PARAMS["weight_decay"])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.TRANSFORMER_PARAMS["n_epochs"])
        loss_fn = nn.MSELoss()

        self.net_.train()

        for _ in range(config.TRANSFORMER_PARAMS["n_epochs"]):
            epoch_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                preds = self.net_(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
        self.is_fitted = True
        return self


    def predict(self,X):
        self._validate_fitted()
        X_tensor=self._to_tensor(X)
        self.net_.eval()
        with torch.no_grad():
            preds=self.net_(X_tensor)

        return preds.cpu().numpy()     # back to numpy
        