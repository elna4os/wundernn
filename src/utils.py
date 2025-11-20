from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def train_val_split(
    df: pd.DataFrame,
    val_size: int,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into training and validation sets based on unique sequence indices.

    Args:
        df (pd.DataFrame): Input dataframe containing a 'seq_ix' column
        val_size (int): Number of unique sequences to include in the validation set
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames for training and validation sets.
    """

    logger.info(f"Splitting to train/val. Dataset size: {len(df)}")
    seq_indices = df['seq_ix'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(seq_indices)

    val_seq_indices = seq_indices[:val_size]
    train_seq_indices = seq_indices[val_size:]

    df_train = df[df['seq_ix'].isin(train_seq_indices)].reset_index(drop=True)
    df_val = df[df['seq_ix'].isin(val_seq_indices)].reset_index(drop=True)
    logger.info(f"Train size: {len(df_train)}, Val size: {len(df_val)}")

    return df_train, df_val
