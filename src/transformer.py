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
        self.embeddings = nn.Linear(1, d_model)
        # (batch, n_features, d_model)

        # learned positional embedding
        self.pos_embedding = nn.Embedding(n_features, d_model)
        self.dropout = nn.Dropout(p=dropout)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = x.unsqueeze(-1)                                      # (batch, n_features, 1)
        x = self.embeddings(x)                                   # (batch, n_features, d_model)

        positions = torch.arange(self.n_features, device=x.device)
        x = x + self.pos_embedding(positions)                    # (batch, n_features, d_model)
        x = self.dropout(x)

        x = self.transformer(x)                                  # (batch, n_features, d_model)
        x = x.mean(dim=1)                                        # (batch, d_model)
        x = self.output_head(x)                                  # (batch, 1)
        return x.squeeze(-1)                                     # (batch,)
