"""
Weather cleaning pipeline (Phase 2 — Step 1).

Turns raw Open-Meteo hourly data into two analysis-ready artefacts:

    data/interim/weather_hourly_clean.csv   # aligned, imputed, quality-flagged
    data/interim/weather_daily_clean.csv    # derived daily aggregates with
                                            # circular-aware wind statistics

Design decisions
----------------
* **Defensive, not destructive.**
  Outliers are flagged with boolean ``*_outlier`` columns; only true physical
  impossibilities (e.g. negative precipitation, humidity > 100%) are clipped.
  The forecasting models get to see the real distribution.

* **Circular statistics for wind direction.**
  Direction is never averaged as a plain scalar. Daily means use a
  speed-weighted unit-vector average: ``dir = atan2(-U_bar, -V_bar)``.

* **Idempotent.**
  Running the pipeline twice produces bitwise-identical output. Column order
  is stable, dropped columns are named explicitly.

* **Alignment first, imputation second.**
  We reindex each city to a complete hourly grid before touching values, so
  downstream lag features can assume a regular cadence.

Public API
----------
- :func:`drop_empty_columns`
- :func:`align_hourly_grid`
- :func:`clip_to_physical_range`
- :func:`interpolate_short_gaps`
- :func:`flag_outliers_monthly_iqr`
- :func:`clean_hourly`
- :func:`resample_to_daily`
- :func:`clean_weather_pipeline` (top-level orchestrator)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import INTERIM_DIR
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
# Hard physical bounds. Values outside are clipped and counted in the log.
PHYSICAL_RANGES: Dict[str, Tuple[float, float]] = {
    "temperature_2m":               (-60.0, 60.0),
    "relative_humidity_2m":         (0.0, 100.0),
    "dew_point_2m":                 (-60.0, 60.0),
    "precipitation":                (0.0, 300.0),
    "rain":                         (0.0, 300.0),
    "cloud_cover":                  (0.0, 100.0),
    "cloud_cover_low":              (0.0, 100.0),
    "cloud_cover_mid":              (0.0, 100.0),
    "cloud_cover_high":             (0.0, 100.0),
    "vapour_pressure_deficit":      (0.0, 20.0),
    "wind_speed_10m":               (0.0, 80.0),
    "wind_direction_10m":           (0.0, 360.0),
    "wind_gusts_10m":               (0.0, 120.0),
    "soil_temperature_0_to_7cm":    (-50.0, 70.0),
    "soil_temperature_7_to_28cm":   (-50.0, 70.0),
    "soil_temperature_28_to_100cm": (-50.0, 70.0),
    "sunshine_duration":            (0.0, 3600.0),
    "shortwave_radiation":          (0.0, 1500.0),
    "direct_radiation":             (0.0, 1500.0),
}

# Variables we'll eventually forecast — outlier flagging is most important for these
FORECAST_CORE_VARS: List[str] = [
    "temperature_2m", "wind_speed_10m", "wind_direction_10m",
    "rain", "precipitation",
]

# Columns that the raw Open-Meteo snapshot systematically returns empty.
# Not a hardcoded drop list — `drop_empty_columns` decides from the data.
KNOWN_EMPTY_HINT: List[str] = [
    "temperature_80m", "evapotranspiration",
    "wind_speed_80m", "wind_direction_80m",
]


# ----------------------------------------------------------------------------
# 1. Column-level cleanup
# ----------------------------------------------------------------------------

def drop_empty_columns(df: pd.DataFrame, threshold: float = 0.999) -> pd.DataFrame:
    """Drop columns whose fraction of missing values exceeds ``threshold``.

    Default 0.999 -> effectively "drop if always NaN". Logs what was removed.
    """
    miss = df.isna().mean()
    to_drop = miss[miss >= threshold].index.tolist()
    if to_drop:
        logger.info("Dropping %d empty columns (miss >= %.1f%%): %s",
                    len(to_drop), threshold * 100, to_drop)
        df = df.drop(columns=to_drop)
    return df


# ----------------------------------------------------------------------------
# 2. Time grid alignment
# ----------------------------------------------------------------------------

def align_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each city to a complete hourly grid spanning its own min/max date.

    If a city currently has a gap (e.g. a 3-hour API outage), this inserts
    rows with NaN values so ``interpolate_short_gaps`` can repair them and
    downstream lag features can assume regular cadence.
    """
    if df["date"].dt.tz is None:
        df = df.copy()
        df["date"] = df["date"].dt.tz_localize("UTC")

    aligned: List[pd.DataFrame] = []
    for city, g in df.groupby("City", sort=False):
        g = g.sort_values("date").drop_duplicates(subset=["date"])
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="h", tz="UTC")
        inserted = len(full_idx) - len(g)
        if inserted:
            logger.info("  %-12s inserting %d missing hourly rows", city, inserted)
        reindexed = (
            g.set_index("date")
             .reindex(full_idx)
             .rename_axis("date")
             .reset_index()
        )
        reindexed["City"] = city
        aligned.append(reindexed)

    out = pd.concat(aligned, ignore_index=True)
    # Preserve leading City, date column order
    cols = ["City", "date"] + [c for c in out.columns if c not in ("City", "date")]
    return out[cols]


# ----------------------------------------------------------------------------
# 3. Physical-range clipping
# ----------------------------------------------------------------------------

def clip_to_physical_range(
    df: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Clip data-entry errors (values outside physically possible bounds).

    Counts are logged per column. This is conservative: the ranges are set so
    that only true impossibilities are touched.
    """
    ranges = ranges or PHYSICAL_RANGES
    df = df.copy()
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        s = df[col]
        below = (s < lo).sum()
        above = (s > hi).sum()
        if below + above:
            logger.warning("Clipping %-28s below=%d above=%d (range %.1f..%.1f)",
                           col, below, above, lo, hi)
            df[col] = s.clip(lower=lo, upper=hi)
    return df


# ----------------------------------------------------------------------------
# 4. Gap interpolation
# ----------------------------------------------------------------------------

def interpolate_short_gaps(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    max_gap_hours: int = 6,
) -> pd.DataFrame:
    """Fill NaN runs of length <= ``max_gap_hours`` by time-linear interpolation.

    Runs longer than the limit are left as NaN (caller can decide whether to
    forward-fill, drop, or flag). Done per city to prevent bleed across
    station boundaries.
    """
    cols = list(columns) if columns is not None else [
        c for c in df.columns if c not in ("City", "date") and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.copy().sort_values(["City", "date"])

    for city, g in df.groupby("City", sort=False):
        idx = g.index
        sub = g.set_index("date")[cols]
        filled = sub.interpolate(
            method="time", limit=max_gap_hours, limit_direction="both", limit_area="inside"
        )
        df.loc[idx, cols] = filled.values

    logger.info("Interpolated short gaps (<=%dh) in %d columns", max_gap_hours, len(cols))
    return df


# ----------------------------------------------------------------------------
# 5. Outlier flagging (non-destructive)
# ----------------------------------------------------------------------------

def flag_outliers_monthly_iqr(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    k: float = 3.0,
) -> pd.DataFrame:
    """Add ``<col>_outlier`` bool columns for values beyond Q1-k·IQR … Q3+k·IQR.

    Statistics are computed per city × per calendar-month so seasonality does
    not inflate the IQR. Default ``k=3`` flags only egregious outliers — tune
    to 1.5 for a tighter filter.
    """
    cols = list(columns) if columns is not None else FORECAST_CORE_VARS
    cols = [c for c in cols if c in df.columns]
    df = df.copy()
    df["_month"] = df["date"].dt.month

    total_flags = 0
    for col in cols:
        flag_col = f"{col}_outlier"
        flags = pd.Series(False, index=df.index)
        for (city, month), g in df.groupby(["City", "_month"], sort=False):
            s = g[col].dropna()
            if len(s) < 30:
                continue
            q1, q3 = np.percentile(s, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            mask = (g[col] < lo) | (g[col] > hi)
            flags.loc[mask[mask].index] = True
        df[flag_col] = flags
        n = int(flags.sum())
        total_flags += n
        logger.info("Flagged %-28s outliers: %d (%.3f%%)",
                    col, n, 100 * n / len(df) if len(df) else 0.0)

    df = df.drop(columns=["_month"])
    logger.info("Total outlier flags across %d columns: %d", len(cols), total_flags)
    return df


# ----------------------------------------------------------------------------
# 6. Hourly cleaning orchestrator
# ----------------------------------------------------------------------------

def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Full hourly-cleaning pipeline.

    Step order matters: we drop empty cols first (so we don't interpolate
    100%-NaN columns), then align the grid (so interpolation has a regular
    index), then clip physical impossibilities, then interpolate short gaps,
    then flag statistical outliers.
    """
    logger.info("--- clean_hourly: input shape %s ---", df.shape)
    df = drop_empty_columns(df)
    df = align_hourly_grid(df)
    df = clip_to_physical_range(df)
    df = interpolate_short_gaps(df)
    df = flag_outliers_monthly_iqr(df, columns=FORECAST_CORE_VARS)

    # Final NaN report
    residual = df.drop(columns=["City", "date"]).isna().mean() * 100
    bad = residual[residual > 0].round(2)
    if len(bad):
        logger.warning("Residual NaNs after cleaning (will be handled in FE):\n%s", bad.to_string())
    else:
        logger.info("Hourly frame is fully dense after cleaning.")

    logger.info("--- clean_hourly: output shape %s ---", df.shape)
    return df


# ----------------------------------------------------------------------------
# 7. Daily resampling with circular statistics
# ----------------------------------------------------------------------------

def _wind_unit_vectors(speed: pd.Series, direction_deg: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Decompose wind into (U, V) meteorological components.

    U = east-west (positive = wind *from* the east), V = north-south.
    With the convention that ``direction_deg`` is the direction the wind is
    coming FROM (meteorological standard).
    """
    rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(rad)
    v = -speed * np.cos(rad)
    return u, v


def _circular_direction_from_uv(u_mean: pd.Series, v_mean: pd.Series) -> pd.Series:
    """Return the meteorological wind direction (degrees, FROM) from mean U,V."""
    return (np.rad2deg(np.arctan2(-u_mean, -v_mean)) + 360.0) % 360.0


def resample_to_daily(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned hourly data into a daily frame suitable for forecasting.

    Aggregation rules:
        - Temperature:          mean, min, max
        - Humidity:             mean
        - Rain, precipitation:  sum (mm/day)
        - Wind speed:           mean, max
        - Wind direction:       speed-weighted circular mean (degrees, FROM)
        - Cloud cover:          mean
        - Soil temperature:     mean
        - Sunshine duration:    sum
        - Shortwave / direct radiation: mean
    """
    logger.info("Resampling to daily granularity ...")
    df = df_hourly.copy()
    df["date_day"] = df["date"].dt.floor("D")

    # Pre-compute wind vectors for speed-weighted circular mean
    if {"wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        u, v = _wind_unit_vectors(df["wind_speed_10m"], df["wind_direction_10m"])
        df["_u"] = u
        df["_v"] = v

    agg: Dict[str, object] = {}
    if "temperature_2m" in df.columns:
        agg["temperature_2m"] = ["mean", "min", "max"]
    if "relative_humidity_2m" in df.columns:
        agg["relative_humidity_2m"] = "mean"
    if "dew_point_2m" in df.columns:
        agg["dew_point_2m"] = "mean"
    if "precipitation" in df.columns:
        agg["precipitation"] = "sum"
    if "rain" in df.columns:
        agg["rain"] = "sum"
    if "cloud_cover" in df.columns:
        agg["cloud_cover"] = "mean"
    for c in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
        if c in df.columns:
            agg[c] = "mean"
    if "vapour_pressure_deficit" in df.columns:
        agg["vapour_pressure_deficit"] = "mean"
    if "wind_speed_10m" in df.columns:
        agg["wind_speed_10m"] = ["mean", "max"]
    if "wind_gusts_10m" in df.columns:
        agg["wind_gusts_10m"] = "max"
    for c in ("soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
              "soil_temperature_28_to_100cm"):
        if c in df.columns:
            agg[c] = "mean"
    if "sunshine_duration" in df.columns:
        agg["sunshine_duration"] = "sum"
    if "shortwave_radiation" in df.columns:
        agg["shortwave_radiation"] = "mean"
    if "direct_radiation" in df.columns:
        agg["direct_radiation"] = "mean"
    if {"_u", "_v"}.issubset(df.columns):
        agg["_u"] = "mean"
        agg["_v"] = "mean"

    daily = df.groupby(["City", "date_day"], as_index=False).agg(agg)

    # Flatten MultiIndex columns
    flat_cols: List[str] = []
    for col in daily.columns:
        if isinstance(col, tuple):
            a, b = col
            if b in ("", None):
                flat_cols.append(a)
            else:
                flat_cols.append(f"{a}_{b}")
        else:
            flat_cols.append(col)
    daily.columns = flat_cols
    daily = daily.rename(columns={"date_day": "date"})

    # Circular wind direction from speed-weighted U, V means
    if {"_u_mean", "_v_mean"}.issubset(daily.columns):
        daily["wind_direction_10m"] = _circular_direction_from_uv(daily["_u_mean"], daily["_v_mean"])
        daily = daily.drop(columns=["_u_mean", "_v_mean"])

    # Sort & reset
    daily = daily.sort_values(["City", "date"]).reset_index(drop=True)
    logger.info("Daily frame: %d rows x %d cols (%d cities, %s..%s)",
                len(daily), daily.shape[1], daily["City"].nunique(),
                daily["date"].min().date(), daily["date"].max().date())
    return daily


# ----------------------------------------------------------------------------
# 8. Top-level orchestrator
# ----------------------------------------------------------------------------

def clean_weather_pipeline(
    hourly_input: Optional[Path] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw hourly weather, clean it, resample to daily, and persist both.

    Returns
    -------
    (hourly_clean, daily_clean)
    """
    hourly_input = hourly_input or (INTERIM_DIR / "weather_hourly_raw.csv")
    if not hourly_input.exists():
        raise FileNotFoundError(
            f"{hourly_input} not found. Run Phase 1 (notebook 01) first."
        )

    logger.info("=" * 72)
    logger.info("PHASE 2.1 - Weather Cleaning starting")
    logger.info("Reading %s", hourly_input)
    logger.info("=" * 72)

    raw = pd.read_csv(hourly_input, parse_dates=["date"])
    if raw["date"].dt.tz is None:
        raw["date"] = raw["date"].dt.tz_localize("UTC")

    hourly_clean = clean_hourly(raw)
    daily_clean = resample_to_daily(hourly_clean)

    if save:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        h_out = INTERIM_DIR / "weather_hourly_clean.csv"
        d_out = INTERIM_DIR / "weather_daily_clean.csv"
        hourly_clean.to_csv(h_out, index=False)
        daily_clean.to_csv(d_out, index=False)
        logger.info("Saved %s rows -> %s (%.2f MB)",
                    f"{len(hourly_clean):,}", h_out, h_out.stat().st_size / 1024 / 1024)
        logger.info("Saved %s rows -> %s (%.2f MB)",
                    f"{len(daily_clean):,}", d_out, d_out.stat().st_size / 1024 / 1024)

    logger.info("=" * 72)
    logger.info("Cleaning complete")
    logger.info("=" * 72)
    return hourly_clean, daily_clean


if __name__ == "__main__":
    clean_weather_pipeline()
