from functools import partial
from pathlib import Path
from typing import Any, Dict

import lightning as L
import pandas as pd
import yaml
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data import StarterDataset, collate_fn
from src.features import prepare_ema_features
from src.models import model_factory


def train_loop(
    dataset_path: str,
    model_class: Any,
    config: Dict[str, Any],
    curr_dir: Path
) -> None:
    """Training loop with K-Fold cross-validation.

    Args:
        dataset_path (str): Path to the dataset
        model_class (Any): Model class to instantiate
        config (Dict[str, Any]): Configuration dictionary
        curr_dir (Path): Current directory path
    """

    out_dir = curr_dir / "logs"

    # Read data
    df = pd.read_parquet(dataset_path)
    feature_cols = df.columns[3:]
    logger.info(f"Dataset shape: {df.shape}")

    # Read params, set global seed
    with open(curr_dir / "params.yaml", "r") as f:
        config = yaml.safe_load(f)

    # EMA features
    df_ema = prepare_ema_features(
        df=df,
        feature_cols=feature_cols,
        alphas=config["additional_features"]["ema"]["alphas"]
    )
    ema_features_cols = [col for col in df_ema.columns if col.startswith("ema_")]
    logger.info(f"EMA features shape: {df_ema.shape}")

    # Merge
    df = pd.merge(
        df,
        df_ema,
        on=["seq_ix", "step_in_seq"],
    )
    logger.info(f"Final dataset shape: {df.shape}")

    # Loop
    seq_ids = df["seq_ix"].unique()
    kf = KFold(n_splits=config["train"]["n_folds"], shuffle=True, random_state=config["seed"])
    for fold, (train_idx, val_idx) in enumerate(kf.split(seq_ids)):
        current_seed = config["seed"] + fold
        logger.info(f"Starting fold {fold} with seed {current_seed}")
        seed_everything(current_seed, workers=True)

        train_df = df[df["seq_ix"].isin(seq_ids[train_idx])].reset_index(drop=True)
        val_df = df[df["seq_ix"].isin(seq_ids[val_idx])].reset_index(drop=True)
        logger.info(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")

        # Initialize datasets
        train_ds = StarterDataset(df=train_df, feature_cols=feature_cols, ema_feature_cols=ema_features_cols)
        val_ds = StarterDataset(df=val_df, feature_cols=feature_cols, ema_feature_cols=ema_features_cols)

        # Initialize dataloaders
        collate_fn_partial = partial(
            collate_fn,
            warmup_steps=config["train"]["model"]["warmup_steps"],
            out_size=config["train"]["model"]["out_size"]
        )
        train_dataloader = DataLoader(
            dataset=train_ds,
            batch_size=config["train"]["dataloader"]["batch_size"],
            shuffle=True,
            num_workers=config["train"]["dataloader"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn_partial,
            persistent_workers=True
        )
        val_dataloader = DataLoader(
            dataset=val_ds,
            batch_size=config["train"]["dataloader"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["dataloader"]["num_workers"],
            pin_memory=True,
            collate_fn=collate_fn_partial,
            persistent_workers=True
        )

        # Create model
        model = model_factory(
            model_class=model_class,
            config=config
        )

        # Trainer
        trainer = L.Trainer(
            accelerator=config["train"]["loop"]["accelerator"],
            max_epochs=config["train"]["loop"]["max_epochs"],
            logger=TensorBoardLogger(out_dir / f"fold_{fold}"),
            callbacks=[EarlyStopping(
                monitor="val/r2",
                min_delta=float(config["train"]["early_stopping"]["min_delta"]),
                patience=config["train"]["early_stopping"]["patience"],
                mode="max"
            )],
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
