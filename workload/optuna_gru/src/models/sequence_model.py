"""Shared GRU/LSTM training implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.models.common import (
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    add_common_args,
    build_model_feature_frame,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)


@dataclass
class ScalingParams:
    mean: np.ndarray
    std: np.ndarray


class SequenceRegressor:
    @staticmethod
    def gru(
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        import torch

        return _TorchSequenceRegressor(torch.nn.GRU, input_size, hidden_size, num_layers, dropout)

    @staticmethod
    def lstm(
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        import torch

        return _TorchSequenceRegressor(torch.nn.LSTM, input_size, hidden_size, num_layers, dropout)


_TorchModuleBase = torch.nn.Module if torch is not None else object


class _TorchSequenceRegressor(_TorchModuleBase):
    def __init__(
        self,
        recurrent_cls,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        import torch

        super().__init__()
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        output, _ = self.recurrent(inputs)
        return self.output(output[:, -1, :]).squeeze(-1)


def parse_args(model_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {model_name.upper()} model.")
    add_common_args(parser)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be a positive integer")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be a positive integer")
    if not 0 < args.learning_rate < 1:
        parser.error("--learning-rate must be greater than 0 and less than 1")
    if args.num_layers <= 0:
        parser.error("--num-layers must be a positive integer")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be greater than or equal to 0 and less than 1")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.gradient_clip < 0:
        parser.error("--gradient-clip must be non-negative")

    return args


def scale_values(values: np.ndarray, params: ScalingParams | None = None) -> tuple[np.ndarray, ScalingParams]:
    if params is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std == 0, 1, std)
        params = ScalingParams(mean=mean, std=std)
    return (values - params.mean) / params.std, params


def make_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= sequence_length:
        raise ValueError("not enough rows to build sequence dataset")

    inputs = []
    targets = []
    for index in range(sequence_length, len(values)):
        inputs.append(values[index - sequence_length : index])
        targets.append(values[index, 0])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def recursive_holdout_forecast(
    model,
    scaled_all: np.ndarray,
    holdout_start: int,
    sequence_length: int,
) -> np.ndarray:
    """Forecast holdout recursively without using true holdout target history."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if len(scaled_all) < sequence_length:
        raise ValueError(
            f"scaled_all must have at least sequence_length rows "
            f"(got {len(scaled_all)} rows and sequence_length={sequence_length})"
        )
    if not isinstance(holdout_start, int):
        raise ValueError("holdout_start must be an integer")
    if holdout_start < sequence_length or holdout_start >= len(scaled_all):
        raise ValueError(
            f"holdout_start must be >= sequence_length and < len(scaled_all) "
            f"(got holdout_start={holdout_start}, sequence_length={sequence_length}, "
            f"len(scaled_all)={len(scaled_all)})"
        )

    import torch

    forecast_values = scaled_all.copy()
    predictions = []
    if hasattr(model, "eval"):
        model.eval()

    for index in range(holdout_start, len(forecast_values)):
        window = forecast_values[index - sequence_length : index]
        with torch.no_grad():
            prediction = model(torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)).item()
        forecast_values[index, 0] = prediction
        predictions.append(prediction)

    return np.asarray(predictions, dtype=np.float32)


def train_sequence_regressor(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int = 0,
    weight_decay: float = 0.0,
    gradient_clip: float = 0.0,
) -> None:
    """Train a sequence regressor with full-batch or mini-batch updates."""
    import torch

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    if hasattr(model, "train"):
        model.train()

    train_x_tensor = torch.tensor(train_x)
    train_y_tensor = torch.tensor(train_y)
    sample_count = len(train_x_tensor)
    effective_batch_size = batch_size if batch_size > 0 else sample_count

    for _ in range(epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, effective_batch_size):
            batch_indices = permutation[start : start + effective_batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(train_x_tensor[batch_indices]), train_y_tensor[batch_indices])
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()


def forecast_sequence_values(
    model,
    train: np.ndarray,
    forecast_frame: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, ScalingParams]:
    """Scale rows, recursively forecast the tail frame, and invert target scaling."""
    _, scaling = scale_values(train)
    combined = np.vstack([train, forecast_frame])
    scaled_combined, _ = scale_values(combined, scaling)
    scaled_predictions = recursive_holdout_forecast(
        model,
        scaled_combined,
        holdout_start=len(train),
        sequence_length=sequence_length,
    )
    target_mean = scaling.mean[0]
    target_std = scaling.std[0]
    return np.maximum((scaled_predictions * target_std) + target_mean, 0), scaling


def main(model_name: str, model_cls: Callable[..., _TorchSequenceRegressor]) -> None:
    import torch

    args = parse_args(model_name)
    torch.manual_seed(42)
    np.random.seed(42)

    df = load_traffic_data(args.data_path)
    model_frame = df[["y"]].join(build_model_feature_frame(df))
    train, holdout = split_train_holdout(df, args.holdout_ratio)
    columns = ["y", *MODEL_FEATURE_COLUMNS]

    train_values = model_frame.iloc[: len(train)][columns].to_numpy(dtype=float)
    all_values = model_frame[columns].to_numpy(dtype=float)
    scaled_train, scaling = scale_values(train_values)
    scaled_all, _ = scale_values(all_values, scaling)

    train_x, train_y = make_sequences(scaled_train, args.sequence_length)
    holdout_start = len(train)

    model = model_cls(
        input_size=len(columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_sequence_regressor(
        model,
        train_x,
        train_y,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
    )

    scaled_predictions = recursive_holdout_forecast(
        model,
        scaled_all,
        holdout_start,
        args.sequence_length,
    )

    target_mean = scaling.mean[0]
    target_std = scaling.std[0]
    predictions_values = np.maximum((scaled_predictions * target_std) + target_mean, 0)

    predictions = build_prediction_frame(holdout, predictions_values, model_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{model_name}.pt")

    metrics = save_model_outputs(
        model_name,
        predictions,
        {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": columns,
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
            "hidden_size": args.hidden_size,
            "learning_rate": args.learning_rate,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "model_path": str((MODELS_DIR / f"{model_name}.pt").relative_to(MODELS_DIR.parent)),
        },
    )
    print(metrics)
