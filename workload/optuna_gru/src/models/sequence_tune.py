"""Tune GRU/LSTM hyperparameters with Optuna for autoscaling metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_predictions
from src.models.common import (
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    RESULTS_DIR,
    add_common_args,
    build_model_feature_frame,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)
from src.models.sequence_model import (
    SequenceRegressor,
    forecast_sequence_values,
    make_sequences,
    scale_values,
    train_sequence_regressor,
)


MODEL_CHOICES = ("gru", "lstm")

if TYPE_CHECKING:
    import optuna


def tuned_model_name(model_name: str) -> str:
    """Return the persisted model name for tuned sequence models."""
    return f"{model_name}_tuned"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sequence model hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Tune GRU/LSTM parameters with Optuna.")
    add_common_args(parser)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True, help="Sequence model to tune.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna trials. Only 1 is supported for deterministic seeded tuning.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Number of CPU threads PyTorch can use per process. 0 keeps the PyTorch default.",
    )
    parser.add_argument(
        "--interop-threads",
        type=int,
        default=0,
        help="Number of PyTorch inter-op CPU threads. 0 keeps the PyTorch default.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna and PyTorch.")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL (e.g. sqlite:////path/study.db or "
            "mysql+pymysql://user:pass@host/db). Omit for a single-process, "
            "in-memory study. Set this to let multiple processes share one "
            "study and split its trials."
        ),
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name to create/join when --storage is set. Defaults to '<model>-tuning'.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="Tail ratio of the training period used for Optuna validation.",
    )
    parser.add_argument(
        "--smape-weight",
        type=float,
        default=0.1,
        help="Penalty weight for SMAPE in the objective score.",
    )
    parser.add_argument(
        "--over-provisioning-weight",
        type=float,
        default=0.2,
        help="Penalty weight for over-provisioning rate in the objective score.",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be a positive integer.")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be a positive integer.")
    if args.n_jobs != 1:
        parser.error("--n-jobs must be 1 because seeded PyTorch trials share process-wide RNG state.")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be non-negative.")
    if args.interop_threads < 0:
        parser.error("--interop-threads must be non-negative.")
    if not 0 < args.validation_ratio < 1:
        parser.error("--validation-ratio must be greater than 0 and less than 1.")
    if args.smape_weight < 0:
        parser.error("--smape-weight must be non-negative.")
    if args.over_provisioning_weight < 0:
        parser.error("--over-provisioning-weight must be non-negative.")

    return args


def configure_torch_cpu_threads(cpu_threads: int, interop_threads: int) -> None:
    """Configure PyTorch CPU thread pools when explicit values are provided."""
    import torch

    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    if interop_threads > 0:
        torch.set_num_interop_threads(interop_threads)


def resolve_model_cls(model_name: str):
    """Map a model name to its sequence regressor factory."""
    if model_name == "gru":
        return SequenceRegressor.gru
    if model_name == "lstm":
        return SequenceRegressor.lstm
    raise ValueError(f"Unsupported sequence model: {model_name}")


def suggest_params(trial: "optuna.Trial") -> dict[str, int | float]:
    """Suggest sequence model parameter values for one Optuna trial."""
    num_layers = trial.suggest_int("num_layers", 1, 3)
    return {
        "sequence_length": trial.suggest_categorical("sequence_length", [12, 24, 48, 72, 168]),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 60),
        "batch_size": trial.suggest_categorical("batch_size", [0, 256, 512, 1024]),
        "num_layers": num_layers,
        "dropout": trial.suggest_float("dropout", 0.0, 0.4) if num_layers > 1 else 0.0,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "gradient_clip": trial.suggest_categorical("gradient_clip", [0.0, 0.5, 1.0, 5.0]),
    }


def normalize_params(params: dict[str, int | float]) -> dict[str, int | float]:
    """Fill conditional Optuna params that may be absent from best_params."""
    normalized = dict(params)
    normalized.setdefault("dropout", 0.0)
    return normalized


def autoscaling_objective_score(
    metrics: dict[str, float],
    smape_weight: float = 0.1,
    over_provisioning_weight: float = 0.2,
) -> float:
    """Score metrics with under-provisioning as the primary objective."""
    return (
        metrics["under_provisioning_rate"]
        + smape_weight * metrics["smape"]
        + over_provisioning_weight * metrics["over_provisioning_rate"]
    )


def evaluate_sequence_params(
    model_name: str,
    train: pd.DataFrame,
    forecast_frame: pd.DataFrame,
    params: dict[str, int | float],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float], object]:
    """Train a GRU/LSTM with params and evaluate recursive forecasts."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    model_cls = resolve_model_cls(model_name)
    params = normalize_params(params)
    columns = ["y", *MODEL_FEATURE_COLUMNS]
    sequence_length = int(params["sequence_length"])

    train_values = train[["y"]].join(build_model_feature_frame(train))[columns].to_numpy(dtype=float)
    forecast_values = forecast_frame[["y"]].join(build_model_feature_frame(forecast_frame))[columns].to_numpy(dtype=float)
    scaled_train, _ = scale_values(train_values)
    train_x, train_y = make_sequences(scaled_train, sequence_length)

    model = model_cls(
        input_size=len(columns),
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params.get("dropout", 0.0)),
    )
    train_sequence_regressor(
        model,
        train_x,
        train_y,
        epochs=int(params["epochs"]),
        learning_rate=float(params["learning_rate"]),
        batch_size=int(params["batch_size"]),
        weight_decay=float(params["weight_decay"]),
        gradient_clip=float(params["gradient_clip"]),
    )

    prediction_values, _ = forecast_sequence_values(
        model,
        train_values,
        forecast_values,
        sequence_length,
    )
    predictions = build_prediction_frame(forecast_frame, prediction_values, tuned_model_name(model_name))
    metrics = evaluate_predictions(predictions["actual"], predictions["predicted"])
    return predictions, metrics, model


def objective_factory(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    smape_weight: float,
    over_provisioning_weight: float,
):
    """Build an Optuna objective function bound to train/validation data."""
    import optuna

    def objective(trial: "optuna.Trial") -> float:
        try:
            params = suggest_params(trial)
            _, metrics, _ = evaluate_sequence_params(
                model_name,
                train,
                validation,
                params,
                seed + trial.number,
            )
            score = autoscaling_objective_score(metrics, smape_weight, over_provisioning_weight)

            for name, value in metrics.items():
                trial.set_user_attr(name, value)
            trial.set_user_attr("objective_score", score)
            return score
        except optuna.TrialPruned:
            raise
        except Exception as error:
            trial.set_user_attr("error", str(error))
            raise

    return objective


def write_tuning_outputs(model_name: str, study: "optuna.Study", best_metrics: dict[str, object]) -> None:
    """Persist tuning trial history, best params, and summary files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = RESULTS_DIR / f"{model_name}_tuning_trials.csv"
    best_params_path = RESULTS_DIR / f"{model_name}_best_params.json"
    summary_path = RESULTS_DIR / f"{model_name}_tuning_summary.json"
    best_params = normalize_params(study.best_params)

    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        trials_path,
        index=False,
    )

    with best_params_path.open("w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2, ensure_ascii=False)

    summary = {
        "model": model_name,
        "tuned_model": tuned_model_name(model_name),
        "best_trial": study.best_trial.number,
        "best_objective_score": study.best_value,
        "best_params_path": str(best_params_path.relative_to(RESULTS_DIR.parent.parent)),
        "trials_path": str(trials_path.relative_to(RESULTS_DIR.parent.parent)),
        "tuned_metrics": best_metrics,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main() -> None:
    """Run GRU/LSTM Optuna tuning and save tuned model evaluation outputs."""
    import optuna
    import torch

    args = parse_args()
    configure_torch_cpu_threads(args.cpu_threads, args.interop_threads)
    df = load_traffic_data(args.data_path)
    train_full, holdout = split_train_holdout(df, args.holdout_ratio)
    train, validation = split_train_holdout(train_full, args.validation_ratio)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.storage:
        # 여러 프로세스(다른 노드의 다른 Slurm 잡)가 같은 study에 합류해서
        # 트라이얼을 나눠 처리하게 함 — 분산 실행 시나리오(ADR-0012).
        study_name = args.study_name or f"{args.model}-tuning"
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective_factory(
            args.model,
            train,
            validation,
            args.seed,
            args.smape_weight,
            args.over_provisioning_weight,
        ),
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        catch=(Exception,),
    )

    best_params = normalize_params(study.best_params)
    best_predictions, best_metric_values, best_model = evaluate_sequence_params(
        args.model,
        train_full,
        holdout,
        best_params,
        args.seed,
    )

    model_name = tuned_model_name(args.model)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{model_name}.pt"
    torch.save(best_model.state_dict(), model_path)

    best_metrics = save_model_outputs(
        model_name,
        best_predictions,
        {
            "base_model": args.model,
            "train_rows": len(train_full),
            "validation_rows": len(validation),
            "holdout_rows": len(holdout),
            "features": ["y", *MODEL_FEATURE_COLUMNS],
            "sequence_params": best_params,
            "objective_score": autoscaling_objective_score(
                best_metric_values,
                args.smape_weight,
                args.over_provisioning_weight,
            ),
            "objective_formula": (
                "under_provisioning_rate "
                f"+ {args.smape_weight} * smape "
                f"+ {args.over_provisioning_weight} * over_provisioning_rate"
            ),
            "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
        },
    )
    write_tuning_outputs(args.model, study, best_metrics)
    print(best_metrics)


if __name__ == "__main__":
    main()
