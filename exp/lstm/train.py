from pathlib import Path

import yaml

from src.models import UnidirLSTM
from src.train import train_loop

if __name__ == "__main__":
    curr_dir = Path(__file__).parent.resolve()
    with open(curr_dir / "params.yaml", "r") as f:
        config = yaml.safe_load(f)
    curr_dir = Path(__file__).parent.resolve()
    train_loop(
        dataset_path="datasets/train.parquet",
        model_class=UnidirLSTM,
        config=config,
        curr_dir=curr_dir
    )
