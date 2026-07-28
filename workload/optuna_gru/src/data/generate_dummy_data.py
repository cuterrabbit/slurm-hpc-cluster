"""Generate synthetic traffic data for model comparison experiments.

This script is based on the traffic pattern generator from the previous
prophet-autoscaler project, but it writes outputs to the new repository layout:

- data/raw/: individual source-like CSV files
- data/processed/: common model input CSV

The default dataset spans five years so Prophet, SARIMA, GRU, and LSTM can learn
multiple yearly cycles and evaluate on an approximately one-year holdout split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BASE_REQUEST_RATE = 100.0
BASE_CPU_UTILIZATION = 50.0

# Crop activity weights model expected BNPL agriculture traffic by month.
# The supported crops are rice, pepper, soybean, garlic, and onion.
CROP_ACTIVITY_WEIGHTS = {
    "rice": {
        1: 0.10,
        2: 0.20,
        3: 0.70,
        4: 0.90,
        5: 1.00,
        6: 0.45,
        7: 0.30,
        8: 0.35,
        9: 0.70,
        10: 0.80,
        11: 0.25,
        12: 0.10,
    },
    "pepper": {
        1: 0.15,
        2: 0.35,
        3: 1.00,
        4: 0.95,
        5: 0.55,
        6: 0.45,
        7: 0.55,
        8: 0.75,
        9: 0.75,
        10: 0.35,
        11: 0.15,
        12: 0.10,
    },
    "soybean": {
        1: 0.05,
        2: 0.10,
        3: 0.20,
        4: 0.30,
        5: 0.60,
        6: 0.90,
        7: 0.45,
        8: 0.35,
        9: 0.45,
        10: 0.80,
        11: 0.65,
        12: 0.10,
    },
    "garlic": {
        1: 0.20,
        2: 0.25,
        3: 0.35,
        4: 0.45,
        5: 0.75,
        6: 0.85,
        7: 0.25,
        8: 0.20,
        9: 0.50,
        10: 0.95,
        11: 0.90,
        12: 0.25,
    },
    "onion": {
        1: 0.15,
        2: 0.20,
        3: 0.45,
        4: 0.65,
        5: 0.90,
        6: 0.85,
        7: 0.25,
        8: 0.20,
        9: 0.75,
        10: 0.95,
        11: 0.45,
        12: 0.15,
    },
}

CROP_TRAFFIC_SHARE = {
    "rice": 0.30,
    "pepper": 0.25,
    "soybean": 0.15,
    "garlic": 0.15,
    "onion": 0.15,
}

WEEKLY_WEIGHT = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.7,
    6: 0.4,
}


@dataclass(frozen=True)
class GeneratedPaths:
    request_rate: Path
    cpu_utilization: Path
    anomaly_events: Path
    processed_traffic: Path


def hourly_weight(hour: int) -> float:
    """Return the synthetic daily traffic multiplier for one hour."""
    if 6 <= hour < 9:
        return 0.90
    if 9 <= hour < 12:
        return 0.60
    if 12 <= hour < 14:
        return 0.30
    if 14 <= hour < 18:
        return 0.55
    if 18 <= hour < 21:
        return 0.85
    return 0.10


def generate_base_pattern(start_date: str, end_date: str, freq: str = "1h") -> pd.DataFrame:
    """Create the base timestamp frame used by all generated datasets."""
    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    if timestamps.empty:
        raise ValueError("date range is empty; check --start, --end, and --freq")

    return pd.DataFrame(
        {
            "ds": timestamps,
            "is_monsoon": 0,
            "typhoon_index": 0.0,
        }
    )


def apply_seasonal_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add monthly, weekly, and hourly traffic pattern multipliers."""
    result = df.copy()
    months = result["ds"].dt.month
    crop_columns = []
    for crop, monthly_weights in CROP_ACTIVITY_WEIGHTS.items():
        column = f"crop_{crop}_activity"
        result[column] = months.map(monthly_weights).astype(float)
        crop_columns.append(column)

    result["crop_activity_score"] = sum(
        result[f"crop_{crop}_activity"] * CROP_TRAFFIC_SHARE[crop]
        for crop in CROP_TRAFFIC_SHARE
    )
    result["_seasonal"] = result["crop_activity_score"]
    result["_weekly"] = result["ds"].dt.dayofweek.map(WEEKLY_WEIGHT)
    result["_daily"] = result["ds"].dt.hour.apply(hourly_weight)
    return result


def apply_weather_patterns(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add monsoon and typhoon regressors used as model features."""
    result = df.copy()
    rng = np.random.default_rng(seed)

    for year in sorted(result["ds"].dt.year.unique()):
        monsoon_mask = (result["ds"] >= f"{year}-06-25") & (
            result["ds"] <= f"{year}-07-25 23:00"
        )
        result.loc[monsoon_mask, "is_monsoon"] = 1

        for _ in range(rng.integers(1, 3)):
            month = rng.integers(8, 10)
            day = rng.integers(1, 26)
            landfall = pd.Timestamp(f"{year}-{month:02d}-{day:02d}")

            for offset, index_value in [(-1, 0.1), (0, 0.9), (1, 0.7), (2, 0.3), (3, 0.1)]:
                target = landfall + pd.Timedelta(days=offset)
                target_mask = result["ds"].dt.date == target.date()
                result.loc[target_mask, "typhoon_index"] = np.maximum(
                    result.loc[target_mask, "typhoon_index"].to_numpy(),
                    index_value,
                )

    return result


def inject_anomalies(df: pd.DataFrame, seed: int = 99) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject repeatable stress-test anomaly windows and return an event table."""
    result = df.copy()
    result["_anomaly_boost"] = 1.0
    rng = np.random.default_rng(seed)
    events: list[dict[str, object]] = []

    years = sorted(result["ds"].dt.year.unique())
    for year in years:
        spring_start = pd.Timestamp(f"{year}-03-10")
        spring_end = pd.Timestamp(f"{year}-03-20 23:00")
        spring_mask = (result["ds"] >= spring_start) & (result["ds"] <= spring_end)
        result.loc[spring_mask, "_anomaly_boost"] *= 1.4
        events.append(
            {
                "scenario": "spring_purchase_spike",
                "year": int(year),
                "start": spring_start,
                "end": spring_end,
                "multiplier": 1.4,
            }
        )

        typhoon_start = pd.Timestamp(f"{year}-09-02")
        typhoon_end = pd.Timestamp(f"{year}-09-04 23:00")
        typhoon_mask = (result["ds"] >= typhoon_start) & (result["ds"] <= typhoon_end)
        result.loc[typhoon_mask, "_anomaly_boost"] *= 1.5
        events.append(
            {
                "scenario": "post_typhoon_recovery_spike",
                "year": int(year),
                "start": typhoon_start,
                "end": typhoon_end,
                "multiplier": 1.5,
            }
        )

    for year in years:
        monsoon_start = pd.Timestamp(f"{year}-07-05")
        monsoon_end = pd.Timestamp(f"{year}-07-15 23:00")
        monsoon_mask = (result["ds"] >= monsoon_start) & (result["ds"] <= monsoon_end)
        result.loc[monsoon_mask, "_anomaly_boost"] *= rng.uniform(0.5, 1.5, monsoon_mask.sum())
        events.append(
            {
                "scenario": "monsoon_volatility",
                "year": int(year),
                "start": monsoon_start,
                "end": monsoon_end,
                "multiplier": "uniform(0.5, 1.5)",
            }
        )

    return result, pd.DataFrame(events)


def build_target(
    df: pd.DataFrame,
    base: float,
    seed: int,
    clip_max: float,
) -> np.ndarray:
    """Build one synthetic target series from shared pattern columns."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, base * 0.03, len(df))

    typhoon_effect = np.where(
        df["typhoon_index"] >= 0.8,
        0.4,
        np.where(df["typhoon_index"] >= 0.3, 1.3, 1.0),
    )

    monsoon_rng = np.random.default_rng(seed + 1)
    monsoon_effect = np.where(
        df["is_monsoon"].to_numpy() == 1,
        monsoon_rng.uniform(0.8, 1.2, len(df)),
        1.0,
    )

    values = (
        base
        * df["_seasonal"].to_numpy()
        * df["_weekly"].to_numpy()
        * df["_daily"].to_numpy()
        * typhoon_effect
        * monsoon_effect
        * df["_anomaly_boost"].to_numpy()
        + noise
    )
    return np.clip(values, 0, clip_max)


def build_datasets(start_date: str, end_date: str, freq: str = "1h") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate raw request/CPU datasets, anomaly events, and common model input."""
    base = generate_base_pattern(start_date, end_date, freq=freq)
    patterned = apply_seasonal_patterns(base)
    patterned = apply_weather_patterns(patterned)
    patterned, anomaly_events = inject_anomalies(patterned)

    request_rate = patterned[["ds", "is_monsoon", "typhoon_index"]].copy()
    request_rate["y"] = build_target(
        patterned,
        base=BASE_REQUEST_RATE,
        seed=42,
        clip_max=500.0,
    )

    cpu_utilization = patterned[["ds", "is_monsoon", "typhoon_index"]].copy()
    cpu_utilization["y"] = build_target(
        patterned,
        base=BASE_CPU_UTILIZATION,
        seed=123,
        clip_max=100.0,
    )

    processed_traffic = request_rate.rename(columns={"y": "request_rate"}).copy()
    processed_traffic["cpu_utilization"] = cpu_utilization["y"]
    processed_traffic["hour"] = processed_traffic["ds"].dt.hour
    processed_traffic["day_of_week"] = processed_traffic["ds"].dt.dayofweek
    processed_traffic["month"] = processed_traffic["ds"].dt.month
    processed_traffic["y"] = processed_traffic["request_rate"]

    return request_rate, cpu_utilization, anomaly_events, processed_traffic


def save_datasets(
    request_rate: pd.DataFrame,
    cpu_utilization: pd.DataFrame,
    anomaly_events: pd.DataFrame,
    processed_traffic: pd.DataFrame,
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> GeneratedPaths:
    """Write generated datasets and return their paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    paths = GeneratedPaths(
        request_rate=raw_dir / "dummy_request_rate.csv",
        cpu_utilization=raw_dir / "dummy_cpu_utilization.csv",
        anomaly_events=raw_dir / "dummy_anomaly_events.csv",
        processed_traffic=processed_dir / "traffic.csv",
    )

    request_rate.to_csv(paths.request_rate, index=False)
    cpu_utilization.to_csv(paths.cpu_utilization, index=False)
    anomaly_events.to_csv(paths.anomaly_events, index=False)
    processed_traffic.to_csv(paths.processed_traffic, index=False)
    return paths


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the synthetic traffic data generator.
    
    Parameters:
        None
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            start (str): Start date string (default "2020-01-01").
            end (str): End date/time string (default "2024-12-31 23:00").
            freq (str): Pandas frequency string for timestamps (default "1h").
    """
    parser = argparse.ArgumentParser(description="Generate synthetic traffic data.")
    parser.add_argument("--start", default="2020-01-01", help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", default="2024-12-31 23:00", help="End date, e.g. 2024-12-31 23:00")
    parser.add_argument("--freq", default="1h", help="Pandas frequency string. Default: 1h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_rate, cpu_utilization, anomaly_events, processed_traffic = build_datasets(
        args.start,
        args.end,
        freq=args.freq,
    )
    paths = save_datasets(request_rate, cpu_utilization, anomaly_events, processed_traffic)

    print("Synthetic traffic data generated.")
    print(f"- request_rate: {paths.request_rate} ({len(request_rate):,} rows)")
    print(f"- cpu_utilization: {paths.cpu_utilization} ({len(cpu_utilization):,} rows)")
    print(f"- anomaly_events: {paths.anomaly_events} ({len(anomaly_events):,} rows)")
    print(f"- processed_traffic: {paths.processed_traffic} ({len(processed_traffic):,} rows)")


if __name__ == "__main__":
    main()
