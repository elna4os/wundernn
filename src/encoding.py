import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embedding_dim: int, wave_length: float = 10000.0):
        """
        Args:
            max_len (int): Maximum sequence length
            embedding_dim (int): Embedding dimension
            wave_length (float, optional): Wave length for sinusoidal functions. Defaults to 10000.0.
        """

        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(wave_length) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim)
            start_idx (int, optional): Starting index for positional encoding. Defaults to 0.

        Returns:
            torch.Tensor: Output tensor with positional embeddings added, shape (batch_size, seq_len, embedding_dim)
        """

        seq_len = x.shape[1]
        pos = self.pe[start_idx:start_idx + seq_len, :].unsqueeze(0)  # (1, seq_len, embedding_dim)

        return x + pos
