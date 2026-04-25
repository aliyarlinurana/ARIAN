"""
Service-layer data access for the FastAPI app.

Encapsulates every CSV read + query as a method on :class:`DataService`.
Separating this from the route handlers means:

* the routes stay declarative (no pandas in ``main.py``)
* every query is unit-testable without the web stack
* adding caching / DB back-end later changes one class, not every endpoint

The service holds its data in memory. For this project's data volumes
(~750 forecast rows, ~11 k climatology rows) that's fine; swap to DuckDB
or Polars if you scale up.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_INTERIM   = PROJECT_ROOT / "data" / "interim"
REPORTS_DIR    = PROJECT_ROOT / "reports" / "climate"

RISK_ORDER = {"low": 0, "moderate": 1, "high": 2, "very_high": 3, "unknown": -1}


@dataclass
class DataService:
    """In-memory cache of every artefact the API needs to answer a request."""

    _weather_long: Optional[pd.DataFrame] = None
    _weather_wide: Optional[pd.DataFrame] = None
    _wildfire_risk: Optional[pd.DataFrame] = None
    _headline_answers: Optional[pd.DataFrame] = None
    _annual_trends: Optional[pd.DataFrame] = None
    _fc_anomalies: Optional[pd.DataFrame] = None
    _climatology_daily: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return all([
            self._weather_long is not None,
            self._wildfire_risk is not None,
        ])

    def load_all(self) -> None:
        """Read every artefact from disk into memory."""
        # --- Weather forecast (long) ---
        p = DATA_PROCESSED / "weather_forecast.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["anchor_date", "forecast_date"])
            for c in ("anchor_date", "forecast_date"):
                if df[c].dt.tz is not None:
                    df[c] = df[c].dt.tz_convert(None).dt.tz_localize(None)
            self._weather_long = df
            # Build wide view
            wide = (
                df.pivot_table(
                    index=["City", "anchor_date", "forecast_date", "horizon_days"],
                    columns="target", values="y_pred",
                )
                .reset_index()
            )
            wide.columns.name = None
            self._weather_wide = wide.sort_values(["City", "forecast_date"]).reset_index(drop=True)
            logger.info("Loaded weather: %d rows long / %d rows wide",
                        len(df), len(wide))
        else:
            logger.warning("weather_forecast.csv missing")

        # --- Wildfire risk ---
        p = DATA_PROCESSED / "wildfire_risk_forecast.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["anchor_date", "forecast_date"])
            for c in ("anchor_date", "forecast_date"):
                if df[c].dt.tz is not None:
                    df[c] = df[c].dt.tz_convert(None).dt.tz_localize(None)
            self._wildfire_risk = df.sort_values(["City", "forecast_date"]).reset_index(drop=True)
            logger.info("Loaded wildfire risk: %d rows", len(df))
        else:
            logger.warning("wildfire_risk_forecast.csv missing")

        # --- Climate reports ---
        for attr, filename in [
            ("_headline_answers", "headline_answers.csv"),
            ("_annual_trends", "annual_trends.csv"),
            ("_fc_anomalies", "forecast_anomalies.csv"),
            ("_climatology_daily", "climatology_daily.csv"),
        ]:
            p = REPORTS_DIR / filename
            if p.exists():
                df = pd.read_csv(p)
                if "forecast_date" in df.columns:
                    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
                    if df["forecast_date"].dt.tz is not None:
                        df["forecast_date"] = df["forecast_date"].dt.tz_localize(None)
                setattr(self, attr, df)
                logger.info("Loaded %s: %d rows", filename, len(df))
            else:
                logger.warning("%s missing", filename)

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in ("_weather_long", "_wildfire_risk", "_headline_answers",
                     "_annual_trends", "_fc_anomalies", "_climatology_daily"):
            df = getattr(self, name)
            out[name.lstrip("_")] = None if df is None else len(df)
        return out

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def meta(self) -> Dict[str, Any]:
        wl = self._require("_weather_long")
        return {
            "cities": sorted(wl["City"].unique().tolist()),
            "anchor_date": wl["anchor_date"].iloc[0].date(),
            "forecast_start": wl["forecast_date"].min().date(),
            "forecast_end": wl["forecast_date"].max().date(),
            "weather_targets": sorted(wl["target"].unique().tolist()),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def weather_long(self, city: Optional[str] = None,
                     target: Optional[str] = None) -> List[Dict[str, Any]]:
        df = self._require("_weather_long")
        if city:
            df = df[df["City"] == city]
        if target:
            df = df[df["target"] == target]
        return [
            {
                "city": row.City,
                "forecast_date": row.forecast_date.date(),
                "horizon_days": int(row.horizon_days),
                "target": row.target,
                "y_pred": float(row.y_pred),
            }
            for row in df.itertuples(index=False)
        ]

    def weather_wide(self, city: Optional[str] = None,
                     horizon_max: int = 30) -> List[Dict[str, Any]]:
        df = self._require("_weather_wide")
        df = df[df["horizon_days"] <= horizon_max]
        if city:
            df = df[df["City"] == city]

        out = []
        for row in df.itertuples(index=False):
            d = {
                "city": row.City,
                "forecast_date": row.forecast_date.date(),
                "horizon_days": int(row.horizon_days),
            }
            for tgt in ("temperature_2m", "wind_speed_10m",
                         "wind_direction_10m", "rain", "precipitation"):
                v = getattr(row, tgt, None)
                d[tgt] = float(v) if v is not None and not pd.isna(v) else None
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Wildfire
    # ------------------------------------------------------------------

    def wildfire(self, city: Optional[str] = None,
                 risk_min: Optional[str] = None) -> List[Dict[str, Any]]:
        df = self._require("_wildfire_risk")
        if city:
            df = df[df["City"] == city]
        if risk_min:
            min_rank = RISK_ORDER.get(risk_min, 0)
            df = df[df["risk_category"].map(RISK_ORDER).fillna(-1) >= min_rank]

        return [
            {
                "city": row.City,
                "forecast_date": row.forecast_date.date(),
                "horizon_days": int(row.horizon_days),
                "fire_probability": float(row.fire_probability),
                "risk_category": row.risk_category,
                "expected_fire_count": float(row.expected_fire_count),
            }
            for row in df.itertuples(index=False)
        ]

    def wildfire_summary(self) -> List[Dict[str, Any]]:
        df = self._require("_wildfire_risk")
        agg = df.groupby("City").agg(
            mean_probability=("fire_probability", "mean"),
            max_probability=("fire_probability", "max"),
            days_high_or_above=("risk_category",
                                lambda s: int(s.isin(["high", "very_high"]).sum())),
            total_expected_count=("expected_fire_count", "sum"),
        ).round(3).reset_index()
        return [
            {
                "city": row.City,
                "mean_probability": float(row.mean_probability),
                "max_probability": float(row.max_probability),
                "days_high_or_above": int(row.days_high_or_above),
                "total_expected_count": float(row.total_expected_count),
            }
            for row in agg.itertuples(index=False)
        ]

    # ------------------------------------------------------------------
    # Insights
    # ------------------------------------------------------------------

    def insights(self) -> List[Dict[str, Any]]:
        """Combine every headline finding into one tidy row per city."""
        ha = self._require("_headline_answers")
        risk_df = self._require("_wildfire_risk")

        cities = sorted(risk_df["City"].unique())
        rows = []
        for city in cities:
            mean_p = float(risk_df[risk_df["City"] == city]["fire_probability"].mean())

            def find_one(question: str) -> str:
                sub = ha[(ha["question"] == question) & (ha["city"] == city)]
                return sub["finding"].iloc[0] if len(sub) else "unknown"

            temp = find_one("Will the next 30 days be hotter than normal?")
            wet  = find_one("Will the next 30 days be wetter or drier than normal?")
            wind = find_one("Is mean wind speed changing?")

            if mean_p >= 0.5:   headline = f"{city}: VERY HIGH wildfire risk next 30 days"
            elif mean_p >= 0.25: headline = f"{city}: HIGH wildfire risk next 30 days"
            elif mean_p >= 0.10: headline = f"{city}: moderate wildfire risk next 30 days"
            else:               headline = f"{city}: low wildfire risk next 30 days"

            rows.append({
                "city": city,
                "next_30d_temperature": temp,
                "next_30d_rainfall": wet,
                "wind_trend": wind,
                "mean_fire_probability": round(mean_p, 3),
                "headline": headline,
            })
        return rows

    def trends(self, city: Optional[str] = None,
               variable: Optional[str] = None) -> List[Dict[str, Any]]:
        df = self._require("_annual_trends")
        if city:
            df = df[df["City"] == city]
        if variable:
            df = df[df["variable"] == variable]
        return df.replace({np.nan: None}).to_dict(orient="records")

    def anomalies(self, city: Optional[str] = None,
                  target: Optional[str] = None) -> List[Dict[str, Any]]:
        df = self._require("_fc_anomalies")
        if city:
            df = df[df["City"] == city]
        if target:
            df = df[df["target"] == target]
        keep = ["City", "forecast_date", "target", "y_pred",
                "clim_mean", "clim_std", "anomaly", "z_score", "classification"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()
        if "forecast_date" in df.columns:
            df["forecast_date"] = df["forecast_date"].dt.date.astype(str)
        return df.replace({np.nan: None}).to_dict(orient="records")

    def climatology(self, city: str, variable: str) -> List[Dict[str, Any]]:
        df = self._require("_climatology_daily")
        q = df[(df["City"] == city) & (df["variable"] == variable)]
        return q.replace({np.nan: None}).to_dict(orient="records")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _require(self, attr: str) -> pd.DataFrame:
        """Raise a clean error if the artefact isn't loaded."""
        df = getattr(self, attr, None)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            raise RuntimeError(f"{attr.lstrip('_')!r} not loaded. Run the pipeline and call "
                               f"POST /admin/refresh.")
        return df
