"""
Weather feature engineering (Phase 2 — Step 2).

Turns the cleaned daily weather frame into a model-ready, feature-rich dataset
suitable for SARIMA / Prophet / XGBoost / LSTM training.

Data leak policy (IMPORTANT)
----------------------------
Every feature derived from the time series is **strictly backward-looking**.
Rolling statistics are computed as ``series.shift(1).rolling(w).agg(...)``,
which means the row at date ``t`` sees only data from ``t-w .. t-1``. We never
include the current day in a feature used to predict the current day.

All shifts / rolls are performed **per city** (``groupby("City").transform()``)
so no information leaks between stations.

Feature families produced
-------------------------
1. **Wind vector decomposition** — (U, V) components + sin/cos of direction.
2. **Lag features** — ``col_lag_{k}`` for k in {1, 2, 3, 7, 14, 30, 365}.
3. **Rolling statistics** — mean/std/min/max over {3, 7, 14, 30} days.
4. **First differences** — ``col_diff_{k}`` for k in {1, 7}.
5. **Calendar features** — day-of-year, day-of-week, month, quarter, weekend.
6. **Fourier features** — sin/cos of 2π·k·doy/365.25 for k in {1, 2, 3}
   (annual + semi-annual + tri-annual harmonics). Optional weekly harmonics.
7. **City indicator** — one-hot encoded city (for pooled models).

Public entry point: :func:`build_weather_features`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import INTERIM_DIR, PROCESSED_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration — the forecast target schema
# ============================================================================
# Maps raw FORECAST_TARGETS (from config.py) to the actual daily columns they
# map to after resampling. The modelling phase will pick which to predict.
DAILY_TARGET_COLUMNS: Dict[str, str] = {
    "temperature_2m":     "temperature_2m_mean",
    "wind_speed_10m":     "wind_speed_10m_mean",
    "wind_direction_10m": "wind_direction_10m",   # handled via U,V below
    "rain":               "rain_sum",
    "precipitation":      "precipitation_sum",
}

# Columns we treat as contextual predictors (subject to lag / rolling).
# Excludes the outlier flag columns and the one-hot encoded city.
DEFAULT_CONTEXT_COLUMNS: List[str] = [
    "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
    "relative_humidity_2m_mean", "dew_point_2m_mean",
    "precipitation_sum", "rain_sum",
    "cloud_cover_mean",
    "vapour_pressure_deficit_mean",
    "wind_speed_10m_mean", "wind_speed_10m_max", "wind_gusts_10m_max",
    "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean",
    "sunshine_duration_sum",
    "shortwave_radiation_mean",
]


@dataclass(frozen=True)
class FeatureConfig:
    """Bundle of hyperparameters controlling feature construction.

    Frozen + typed so configurations are reproducible and comparable across
    experiments.
    """

    lags: Sequence[int] = (1, 2, 3, 7, 14, 30, 365)
    rolling_windows: Sequence[int] = (3, 7, 14, 30)
    rolling_aggs: Sequence[str] = ("mean", "std", "min", "max")
    diff_periods: Sequence[int] = (1, 7)
    fourier_periods: Sequence[float] = (365.25,)           # weekly is optional
    fourier_harmonics: int = 3
    include_weekly_fourier: bool = False
    context_columns: Sequence[str] = field(default_factory=lambda: tuple(DEFAULT_CONTEXT_COLUMNS))
    include_city_dummies: bool = True
    group_col: str = "City"
    date_col: str = "date"


# ============================================================================
# 1. Wind vector features
# ============================================================================

def add_wind_vector_features(
    df: pd.DataFrame,
    speed_col: str = "wind_speed_10m_mean",
    direction_col: str = "wind_direction_10m",
) -> pd.DataFrame:
    """Add ``U``, ``V`` wind components and sin/cos of direction.

    * ``U = -speed * sin(dir)`` — east-west component (positive = from east)
    * ``V = -speed * cos(dir)`` — north-south component (positive = from north)

    Meteorological convention: ``direction`` is the compass bearing the wind
    is coming FROM (0° = north, 90° = east).

    sin/cos of the raw direction are also added so non-linear tree models can
    use them directly.
    """
    if speed_col not in df.columns or direction_col not in df.columns:
        logger.warning("Wind columns missing (%s, %s) — skipping vector features",
                       speed_col, direction_col)
        return df

    df = df.copy()
    rad = np.deg2rad(df[direction_col])
    df["wind_u"] = -df[speed_col] * np.sin(rad)
    df["wind_v"] = -df[speed_col] * np.cos(rad)
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    logger.info("Added wind vector features: wind_u, wind_v, wind_dir_sin, wind_dir_cos")
    return df


# ============================================================================
# 2. Lag features
# ============================================================================

def add_lag_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
    group_col: str = "City",
) -> pd.DataFrame:
    """Append ``<col>_lag_<k>`` for every (col, k) combination.

    Leak-safe: uses ``groupby(group_col)[col].shift(k)``. Builds every new
    column in a dict and concatenates once to avoid pandas fragmentation.
    """
    cols = [c for c in columns if c in df.columns]
    lags_list = list(lags)
    grouped = df.groupby(group_col, sort=False)
    new_cols = {
        f"{col}_lag_{k}": grouped[col].shift(k)
        for col in cols for k in lags_list
    }
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added %d lag features (%d cols x %d lags)",
                len(new_cols), len(cols), len(lags_list))
    return out


# ============================================================================
# 3. Rolling statistics
# ============================================================================

_ROLLING_FN = {
    "mean": lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean(),
    "std":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).std(),
    "min":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).min(),
    "max":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).max(),
    "sum":  lambda s, w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).sum(),
}


def add_rolling_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
    aggs: Iterable[str] = ("mean", "std"),
    group_col: str = "City",
) -> pd.DataFrame:
    """Append rolling statistics that use strictly past data.

    For every (col, window, agg) we compute
    ``series.shift(1).rolling(window).agg()`` per group, which guarantees the
    row at date ``t`` sees only data from ``t-w .. t-1``. Built as a single
    ``pd.concat`` to avoid DataFrame fragmentation.
    """
    cols = [c for c in columns if c in df.columns]
    windows_list = list(windows)
    aggs_list = list(aggs)
    for a in aggs_list:
        if a not in _ROLLING_FN:
            raise ValueError(f"Unsupported rolling agg: {a!r}")

    new_cols: Dict[str, pd.Series] = {}
    for col in cols:
        for w in windows_list:
            for agg in aggs_list:
                fn = _ROLLING_FN[agg]
                new_cols[f"{col}_roll{w}_{agg}"] = (
                    df.groupby(group_col, sort=False)[col]
                      .transform(lambda s, fn=fn, w=w: fn(s, w))
                )

    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added %d rolling features (%d cols x %d windows x %d aggs)",
                len(new_cols), len(cols), len(windows_list), len(aggs_list))
    return out


# ============================================================================
# 4. Differenced features
# ============================================================================

def add_diff_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    periods: Iterable[int] = (1, 7),
    group_col: str = "City",
) -> pd.DataFrame:
    """Append ``<col>_diff_<p>`` = x(t) - x(t-p) per city."""
    cols = [c for c in columns if c in df.columns]
    periods_list = list(periods)
    grouped = df.groupby(group_col, sort=False)
    new_cols = {
        f"{col}_diff_{p}": grouped[col].diff(p)
        for col in cols for p in periods_list
    }
    out = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    logger.info("Added %d differenced features", len(new_cols))
    return out


# ============================================================================
# 5. Calendar features
# ============================================================================

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add day-of-year, day-of-week, month, quarter, weekend indicator."""
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["day_of_year"] = d.dt.dayofyear.astype(np.int16)
    df["day_of_week"] = d.dt.dayofweek.astype(np.int8)
    df["month"]       = d.dt.month.astype(np.int8)
    df["quarter"]     = d.dt.quarter.astype(np.int8)
    df["is_weekend"]  = (d.dt.dayofweek >= 5).astype(np.int8)
    df["year"]        = d.dt.year.astype(np.int16)
    logger.info("Added 6 calendar features (day_of_year, day_of_week, month, quarter, is_weekend, year)")
    return df


# ============================================================================
# 6. Fourier features
# ============================================================================

def add_fourier_features(
    df: pd.DataFrame,
    date_col: str = "date",
    period_days: float = 365.25,
    n_harmonics: int = 3,
    prefix: str = "annual",
) -> pd.DataFrame:
    """Append ``{prefix}_sin_k``, ``{prefix}_cos_k`` for k=1..n_harmonics.

    The argument is ``2π · k · day_number / period_days``. With
    ``period_days=365.25`` this yields a smooth continuous encoding of the
    annual cycle that's far better than one-hot month for tree / linear models.
    """
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    # Use day-of-year-like continuous index so leap years don't break phase
    day_num = (d - d.min()).dt.total_seconds() / 86400.0
    for k in range(1, n_harmonics + 1):
        arg = 2.0 * np.pi * k * day_num / period_days
        df[f"{prefix}_sin_{k}"] = np.sin(arg).astype(np.float32)
        df[f"{prefix}_cos_{k}"] = np.cos(arg).astype(np.float32)
    logger.info("Added %d Fourier features (period=%.2f days, %d harmonics)",
                2 * n_harmonics, period_days, n_harmonics)
    return df


# ============================================================================
# 7. City dummies
# ============================================================================

def add_city_dummies(df: pd.DataFrame, group_col: str = "City") -> pd.DataFrame:
    """Append one-hot dummies ``city_<Name>`` for pooled (cross-city) models."""
    if group_col not in df.columns:
        logger.warning("Group column %s missing — skipping city dummies", group_col)
        return df
    dummies = pd.get_dummies(df[group_col], prefix="city", dtype=np.int8)
    df = pd.concat([df, dummies], axis=1)
    logger.info("Added %d city dummy columns: %s",
                dummies.shape[1], list(dummies.columns))
    return df


# ============================================================================
# 8. Orchestrator
# ============================================================================

def build_weather_features(
    df: Optional[pd.DataFrame] = None,
    config: Optional[FeatureConfig] = None,
    save: bool = True,
    output_name: str = "weather_features",
) -> pd.DataFrame:
    """End-to-end feature construction.

    Parameters
    ----------
    df
        Cleaned daily frame. If ``None``, read from ``weather_daily_clean.csv``.
    config
        A :class:`FeatureConfig`. Default uses ``FeatureConfig()``.
    save
        If ``True``, persist to ``data/processed/<output_name>.csv``.

    Returns
    -------
    The feature-augmented DataFrame (sorted by City, date).
    """
    cfg = config or FeatureConfig()
    if df is None:
        p = INTERIM_DIR / "weather_daily_clean.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run notebook 02 first")
        df = pd.read_csv(p, parse_dates=[cfg.date_col])

    if df[cfg.date_col].dt.tz is not None:
        df = df.copy()
        df[cfg.date_col] = df[cfg.date_col].dt.tz_convert("UTC").dt.tz_localize(None)

    logger.info("=" * 72)
    logger.info("PHASE 2.2 - Feature Engineering starting")
    logger.info("Input shape: %s, %d cities", df.shape, df[cfg.group_col].nunique())
    logger.info("=" * 72)

    # Must be sorted before any shift/roll to ensure correct lag semantics
    df = df.sort_values([cfg.group_col, cfg.date_col]).reset_index(drop=True)

    # --- 1. Wind vectors (added first so we can lag/roll the U,V components too) ---
    df = add_wind_vector_features(df)

    # Extend context columns to include the newly-created wind components
    ctx = list(cfg.context_columns)
    for extra in ("wind_u", "wind_v"):
        if extra in df.columns and extra not in ctx:
            ctx.append(extra)

    # --- 2. Lag features ---
    df = add_lag_features(df, ctx, cfg.lags, group_col=cfg.group_col)

    # --- 3. Rolling statistics ---
    df = add_rolling_features(df, ctx, cfg.rolling_windows, cfg.rolling_aggs,
                              group_col=cfg.group_col)

    # --- 4. Differences ---
    df = add_diff_features(df, ctx, cfg.diff_periods, group_col=cfg.group_col)

    # --- 5. Calendar ---
    df = add_calendar_features(df, date_col=cfg.date_col)

    # --- 6. Fourier ---
    df = add_fourier_features(df, date_col=cfg.date_col,
                              period_days=365.25,
                              n_harmonics=cfg.fourier_harmonics,
                              prefix="annual")
    if cfg.include_weekly_fourier:
        df = add_fourier_features(df, date_col=cfg.date_col,
                                  period_days=7.0, n_harmonics=2,
                                  prefix="weekly")

    # --- 7. City dummies ---
    if cfg.include_city_dummies:
        df = add_city_dummies(df, group_col=cfg.group_col)

    added = df.shape[1]
    logger.info("Output shape: %s (%d features total)", df.shape, added)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / f"{output_name}.csv"
        df.to_csv(out, index=False)
        logger.info("Saved %s rows -> %s (%.2f MB)",
                    f"{len(df):,}", out, out.stat().st_size / 1024 / 1024)

    logger.info("=" * 72)
    logger.info("Feature engineering complete")
    logger.info("=" * 72)
    return df


# ============================================================================
# 9. Modelling helpers
# ============================================================================

def list_target_columns() -> Dict[str, str]:
    """Return the forecast target -> daily-column mapping."""
    return dict(DAILY_TARGET_COLUMNS)


def time_train_test_split(
    df: pd.DataFrame,
    test_start: str,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean time-based split: everything before ``test_start`` is train.

    Avoids the classic shuffle-split mistake that leaks future info.
    """
    ts = pd.Timestamp(test_start)
    date = pd.to_datetime(df[date_col])
    train = df[date < ts].copy()
    test  = df[date >= ts].copy()
    logger.info("Split %s -> train=%d rows, test=%d rows",
                test_start, len(train), len(test))
    return train, test


def feature_columns(df: pd.DataFrame, exclude_targets: bool = True) -> List[str]:
    """Return the canonical list of predictor columns (everything except
    meta + the raw daily target columns).
    """
    meta = {"City", "date"}
    outlier_flags = {c for c in df.columns if c.endswith("_outlier")}
    targets = set(DAILY_TARGET_COLUMNS.values()) if exclude_targets else set()
    out = [c for c in df.columns if c not in meta | outlier_flags | targets]
    return out


if __name__ == "__main__":
    build_weather_features()
