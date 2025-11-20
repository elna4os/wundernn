from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def collate_fn(
    batch: List[Dict[str, Any]],
    warmup_steps: int = 100,
    out_size: int = 32
) -> Dict[str, Any]:
    """Collate function for DataLoader

    Args:
        batch (List[Dict[str, Any]]): Initial batch
        warmup_steps (int, optional): Number of warmup steps. Defaults to 100.
        out_size (int, optional): Output size. Defaults to 32.

    Returns:
        Dict[str, Any]: Collated batch
    """

    data = torch.tensor(np.stack([b["features"] for b in batch])).float()
    ema_data = torch.tensor(np.stack([b["ema_features"] for b in batch])).float()

    return {
        "features": data,
        "ema_features": ema_data,
        "targets": data[:, warmup_steps + 1:, :out_size]
    }


class StarterDataset(Dataset):
    """Dataset for LSTM starter experiment
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ema_feature_cols: List[str],
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame with market data
            feature_cols (List[str]): List of time series column names
            ema_feature_cols (List[str]): List of EMA feature column names
        """

        self.df = df
        self.seq_indices = df["seq_ix"].unique()
        self.feature_cols = feature_cols
        self.ema_feature_cols = ema_feature_cols

    def __getitem__(self, index: int) -> Dict[str, Any]:
        seq_ix = self.seq_indices[index]
        seq_data = self.df[self.df["seq_ix"] == seq_ix]

        return {
            "features": seq_data[self.feature_cols].values,
            "ema_features": seq_data[self.ema_feature_cols].values,
        }

    def __len__(self) -> int:
        return len(self.seq_indices)
