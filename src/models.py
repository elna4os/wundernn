from abc import ABC, abstractmethod
from typing import Any, Dict

import lightning as L
import torch
from loguru import logger

from src.encoding import PositionalEncoding
from src.losses import mse
from src.metrcis import r2


class AffineFeaturesFusion(torch.nn.Module):
    """Simple affine fusion layer: F(x) = gamma(z) * x + beta(z)
    """

    def __init__(self, x_dim: int, z_dim: int):
        """
        Args:
            x_dim (int): Main input dimension
            z_dim (int): Conditioning input dimension
        """

        super().__init__()
        self.gamma = torch.nn.Linear(z_dim, x_dim)
        self.beta = torch.nn.Linear(z_dim, x_dim)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(z)
        beta = self.beta(z)

        return gamma * x + beta


class ModelBase(L.LightningModule, ABC):
    """Base class for PyTorch Lightning models.
    """

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Each model must implement the forward method.

        Args:
            batch (Dict[str, Any]): Batch data

        Returns:
            torch.Tensor: Model predictions (batch_size, seq_len, output_size)
        """

        raise NotImplementedError

    def step(self, batch: Dict[str, Any], batch_idx: int, stage: str) -> torch.Tensor:
        """Common step for training and validation.

        Args:
            batch (Dict[str, Any]): Batch data
            batch_idx (int): Batch index (not used)
            stage (str): Stage name (for example, "train" or "val")

        Returns:
            torch.Tensor: Computed loss
        """

        targets = batch["targets"]
        # Ignore warmup steps in loss calculation
        preds = self.forward(batch)[:, self.warmup_steps:-1, :]
        if targets.shape != preds.shape:
            raise ValueError(f"Targets shape {targets.shape} does not match predictions shape {preds.shape}")
        loss = mse(preds=preds, targets=targets)
        metric = r2(preds=preds, targets=targets)
        self.log(f"{stage}/mse", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/r2", metric, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, stage="val")


class UnidirLSTM(ModelBase):
    """Architecture:
    - Input linear projection
    - Fusion with EMA features
    - Positional Encoding
    - Unidirectional LSTM
    - Output linear layer
    """

    def __init__(
        self,
        in_size: int,
        in_ema_size: int,
        in_proj: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        out_size: int = 32,
        warmup_steps: int = 100,
        max_len: int = 1000
    ):
        """
        Args:
            in_size (int): Input feature size
            in_ema_size (int): Input EMA feature size
            in_proj (int): Input projection size
            hidden_size (int): Hidden layer size
            num_layers (int, optional): Number of LSTM layers. Defaults to 1.
            dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            out_size (int, optional): Output feature size. Defaults to 32.
            warmup_steps (int, optional): Number of warmup steps to initialize LSTM. Defaults to 100.
            max_len (int, optional): Maximum sequence length for positional encoding. Defaults to 1000.
        """

        super().__init__()
        self.in_proj_linear = torch.nn.Linear(
            in_features=in_size,
            out_features=in_proj
        )
        self.ema_fusion = AffineFeaturesFusion(
            x_dim=in_proj,
            z_dim=in_ema_size
        )
        self.pos_encoding = PositionalEncoding(
            max_len=max_len,
            embedding_dim=in_proj
        )
        self.lstm = torch.nn.LSTM(
            input_size=in_proj,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        self.linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=out_size
        )
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()
        # Store hidden/cell states explicitly for the inference stage
        self.hidden_state = None
        self.cell_state = None

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        x = self.in_proj_linear(batch["features"])
        x = self.ema_fusion(x, batch["ema_features"])
        x = self.pos_encoding(x)
        lstm_out, _ = self.lstm(x)
        preds = self.linear(lstm_out)

        return preds

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = self.in_proj_linear(batch["features"])
        x = self.ema_fusion(x, batch["ema_features"])
        seq_len = x.shape[1]
        if seq_len == 1:
            # For single-step input, provide the step index for positional encoding
            x = self.pos_encoding(x, start_idx=batch["step_in_seq"])
        else:
            # For multi-step input, use standard positional encoding (first step after "warmup")
            x = self.pos_encoding(x)
        if self.hidden_state is None:
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(x)
        else:
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
                x,
                (self.hidden_state, self.cell_state)
            )
        preds = self.linear(lstm_out)

        return preds

    def clear_states(self):
        """During the inference, when sequence index changes, we need to clear the hidden and cell states
        """

        self.hidden_state = None
        self.cell_state = None


def model_factory(
    model_class: Any,
    config: Dict[str, Any]
) -> Any:
    """Creates model instance based on the provided class and configuration.

    Args:
        model_class (Any): Model class to instantiate
        config (Dict[str, Any]): Configuration dictionary

    Returns:
        Any: Instantiated model instance
    """

    if model_class is UnidirLSTM:
        model = UnidirLSTM(
            in_size=config["train"]["model"]["in_size"],
            in_ema_size=config["train"]["model"]["in_ema_size"],
            in_proj=config["train"]["model"]["in_proj"],
            hidden_size=config["train"]["model"]["hidden_size"],
            num_layers=config["train"]["model"]["num_layers"],
            dropout=config["train"]["model"]["dropout"],
            lr=float(config["train"]["optimizer"]["lr"]),
            out_size=config["train"]["model"]["out_size"],
            warmup_steps=config["train"]["model"]["warmup_steps"]
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    logger.info(f"Created model: {type(model).__name__}")

    return model
