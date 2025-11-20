import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.features import prepare_ema_features
from src.models import UnidirLSTM
from src.utils import DataPoint


class PredictionModel:
    def __init__(self):
        # Data description
        self.ema_alphas = [0.05]
        self.feature_cols = [str(i) for i in range(32)]
        self.ema_cols = []
        for alpha in self.ema_alphas:
            alpha_cols = [f"ema_{alpha}_{col}" for col in self.feature_cols]
            self.ema_cols.extend(alpha_cols)
        self.prev_ema = None

        # Checkpoints
        folds_dir = "logs"
        internal_dir = "lightning_logs/version_0/checkpoints"
        self.models = []
        for curr_fold_dir in os.listdir(folds_dir):
            curr_dir = os.path.join(folds_dir, curr_fold_dir, internal_dir)
            ckpt_filename = os.listdir(curr_dir)[-1]
            ckpt_path = os.path.join(curr_dir, ckpt_filename)
            model = UnidirLSTM.load_from_checkpoint(
                ckpt_path,
                map_location="cpu"
            ).eval()
            self.models.append(model)

        # Sequence data
        self.current_seq_ix = None
        self.sequence_history = []

    @torch.no_grad()
    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset states if new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
            for model in self.models:
                model.clear_states()
            self.prev_ema = None
        # Append current state to history
        self.sequence_history.append(data_point.state.copy())
        if not data_point.need_prediction:
            return None

        # Prepare data
        if self.models[0].hidden_state is None:
            sequences_np = np.stack(self.sequence_history)  # (seq_len, n_features)
            features = torch.tensor(sequences_np).unsqueeze(0).float()  # (1, seq_len, n_features)
            df = pd.DataFrame(
                data=sequences_np,
                columns=self.feature_cols
            )
            df["seq_ix"] = data_point.seq_ix
            df["step_in_seq"] = np.arange(len(df))
            df_ema = prepare_ema_features(
                df=df,
                feature_cols=self.feature_cols,
                alphas=self.ema_alphas
            )
            ema_features = torch.tensor(df_ema[self.ema_cols].values).unsqueeze(0).float()  # (1, seq_len, n_ema_features)
            self.prev_ema = ema_features.squeeze(0)[-1]  # (n_ema_features,)
        else:
            features = torch.tensor(data_point.state).unsqueeze(0).unsqueeze(0).float()  # (1, 1, n_features)
            ema_features = []
            n_features = len(self.feature_cols)
            for i, alpha in enumerate(self.ema_alphas):
                curr_alpha_ema = alpha * features.flatten() + (1 - alpha) * self.prev_ema[i * n_features:(i + 1) * n_features]
                ema_features.append(curr_alpha_ema)
            ema_features = torch.cat(ema_features, dim=0).unsqueeze(0).unsqueeze(0).float()  # (1, 1, n_ema_features)
            self.prev_ema = ema_features[0, 0, :]
        batch = {
            "features": features,
            "ema_features": ema_features,
            "step_in_seq": int(data_point.step_in_seq)
        }

        # Predict
        predictions = []
        for model in self.models:
            pred = model.predict_step(batch, 0)  # (1, seq_len, n_targets)
            last_timestamp_pred = pred[0, -1, :]  # (n_targets,)
            predictions.append(last_timestamp_pred)
        res = torch.mean(torch.stack(predictions), dim=0).numpy()

        return res


if __name__ == "__main__":
    # Only for local testing
    df = pd.read_parquet("datasets/train.parquet")
    model = PredictionModel()
    for row in tqdm(df.values):
        data_point = DataPoint(
            seq_ix=row[0],
            step_in_seq=row[1],
            need_prediction=row[2],
            state=row[3:],
        )
        prediction = model.predict(data_point)
