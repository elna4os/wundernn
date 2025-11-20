from typing import List

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from tqdm import tqdm


def ema(
    data: pd.DataFrame,
    feature_cols: List[str],
    alpha: float
) -> pd.DataFrame:
    """Applies Exponential Moving Average (EMA) to each column in the DataFrame

    Args:
        data (pd.DataFrame): Input DataFrame
        feature_cols (List[str]): List of feature column names to apply EMA on
        alpha (float): Smoothing factor for EMA

    Returns:
        pd.DataFrame: DataFrame containing EMA features. New columns are named as "ema_{alpha}_{original_column_name}"
    """

    ema_df = data[feature_cols].ewm(alpha=alpha, adjust=False).mean()
    ema_df = ema_df.add_prefix(f"ema_{alpha}_")

    return ema_df


def prepare_ema_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    alphas: List[float]
) -> pd.DataFrame:
    """Prepares EMA features for each sequence in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing sequences identified by 'seq_ix'
        feature_cols (List[str]): List of feature column names to apply EMA on
        alphas (List[float]): List of smoothing factors for EMA

    Returns:
        pd.DataFrame: DataFrame containing all EMA features for each sequence
    """

    sequences_data = []
    for _, group in tqdm(df.groupby("seq_ix"), total=df["seq_ix"].nunique()):
        curr_group_data = []
        for alpha in alphas:
            df_curr = ema(
                data=group,
                feature_cols=feature_cols,
                alpha=alpha
            )
            curr_group_data.append(df_curr)
        df_group_ema = pd.concat(curr_group_data, axis=1)
        df_group_ema[['seq_ix', 'step_in_seq']] = group[['seq_ix', 'step_in_seq']]
        sequences_data.append(df_group_ema)

    return pd.concat(sequences_data, axis=0).reset_index(drop=True)


def spectral_entropy(signal: np.ndarray) -> float:
    """Calculates the spectral entropy of a 1D signal using Discrete Cosine Transform (DCT)

    Args:
        signal (np.ndarray): Input 1D signal

    Returns:
        float: Spectral entropy of the signal
    """

    # Compute DCT and power spectrum
    spec = dct(signal) ** 2
    # Normalize the power spectrum to get a probability distribution
    p = spec / np.sum(spec)
    # Compute spectral entropy
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    return entropy


def prepare_se_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_sizes: List[int],
    min_periods: int = 1
) -> pd.DataFrame:
    """Prepares spectral entropy features for each sequence in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing sequences identified by 'seq_ix'
        feature_cols (List[str]): List of feature column names to compute spectral entropy on
        window_sizes (List[int]): List of window sizes for computing spectral entropy
        min_periods (int): Minimum number of observations in window required to have a value. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing all spectral entropy features for each sequence
    """

    sequences_data = []
    for _, group in tqdm(df.groupby("seq_ix"), total=df["seq_ix"].nunique()):
        curr_group_data = []
        for window in window_sizes:
            df_curr = group[feature_cols].rolling(window=window, min_periods=min_periods).apply(
                spectral_entropy,
                raw=True
            )
            df_curr = df_curr.add_prefix(f"se_{window}_")
            curr_group_data.append(df_curr.reset_index(drop=True))
        df_group_se = pd.concat(curr_group_data, axis=1)
        df_group_se[['seq_ix', 'step_in_seq']] = group[['seq_ix', 'step_in_seq']]
        sequences_data.append(df_group_se)

    return pd.concat(sequences_data, axis=0).reset_index(drop=True)
