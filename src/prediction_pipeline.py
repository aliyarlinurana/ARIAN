"""
ARIAN 30-day wildfire risk prediction pipeline.

This script trains several forecast-compatible wildfire classifiers using a
strict temporal split, selects the best calibrated probability model, scores the
30-day weather forecast, and writes dashboard-ready CSV/JSON outputs.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from pandas.errors import PerformanceWarning

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

from src.config import ENG_DAILY, FORECAST_30D, MODELS_F, OUTPUTS, RANDOM_SEED
from src.features import add_calendar_features, add_wildfire_weather_features, compute_fwi_proxy

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")


TRAIN_END = pd.Timestamp("2024-01-01")
TEST_START = pd.Timestamp("2025-01-01")
TARGET = "Fire_Occurred"
RISK_ORDER = ["Low", "Moderate", "High", "Extreme"]
RISK_COLORS = {
    "Low": "#3FA773",
    "Moderate": "#D8A31D",
    "High": "#D96C3B",
    "Extreme": "#B73333",
}

BASE_WEATHER = [
    "Temperature_C_mean",
    "Humidity_percent_mean",
    "Rain_mm_sum",
    "Wind_Speed_kmh_mean",
    "Pressure_hPa_mean",
    "Solar_Radiation_Wm2_mean",
    "Soil_Temp_C_mean",
    "Soil_Moisture_mean",
]
STATIC_FEATURES = [
    "Latitude",
    "Longitude",
    "Elevation",
    "Slope",
    "Trees_pct",
    "Urban_pct",
    "Pop_Total",
    "NDBI",
    "NDVI",
    "EVI",
]
DROP_COLUMNS = {
    "Date",
    "City",
    TARGET,
    "fire_count",
    "mean_brightness",
    "max_frp",
    "Burned_Area_hectares",
}


@dataclass
class ModelResult:
    name: str
    estimator: Pipeline
    calibrator: IsotonicRegression
    threshold: float
    metrics: Dict[str, float]


def _risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "Extreme"
    if probability >= 0.35:
        return "High"
    if probability >= 0.15:
        return "Moderate"
    return "Low"


def _confidence(probability: float) -> float:
    """Readable confidence proxy: distance from the uncertain middle."""
    return float(np.clip(0.55 + abs(probability - 0.5) * 0.8, 0.55, 0.95))


def _climate_summary(row: pd.Series) -> str:
    temp = row.get("Temperature_C_mean", np.nan)
    wind = row.get("Wind_Speed_kmh_mean", np.nan)
    humidity = row.get("Humidity_percent_mean", np.nan)
    rain = row.get("Rain_mm_sum", np.nan)
    fragments = []
    if pd.notna(temp) and temp >= 28:
        fragments.append("hot conditions")
    elif pd.notna(temp) and temp <= 12:
        fragments.append("cool conditions")
    else:
        fragments.append("mild temperatures")
    if pd.notna(wind) and wind >= 18:
        fragments.append("strong wind")
    if pd.notna(humidity) and humidity <= 40:
        fragments.append("dry air")
    if pd.notna(rain) and rain >= 2:
        fragments.append("recent rainfall")
    return ", ".join(fragments).capitalize() + "."


def _warning_text(row: pd.Series) -> str:
    if row["risk_level"] in {"High", "Extreme"}:
        return "High temperature, dry air, and wind can accelerate wildfire spread."
    if row.get("Wind_Speed_kmh_mean", 0) >= 18:
        return "Wind is elevated, so small ignitions could spread faster."
    if row.get("Humidity_percent_mean", 100) <= 40:
        return "Low humidity can dry vegetation and raise ignition sensitivity."
    return "Current conditions suggest limited short-term wildfire pressure."


def _add_lag_roll_features(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    df = df.sort_values(["City", "Date"]).copy()
    for var in variables:
        if var not in df.columns:
            continue
        grouped = df.groupby("City", group_keys=False)[var]
        for lag in [1, 2, 3, 5, 7, 14, 30]:
            df[f"{var}_lag{lag}"] = grouped.shift(lag)
        shifted = grouped.shift(1)
        for window in [3, 7, 14, 30]:
            df[f"{var}_roll{window}_mean"] = shifted.groupby(df["City"]).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f"{var}_roll{window}_std"] = shifted.groupby(df["City"]).rolling(window, min_periods=2).std().reset_index(level=0, drop=True)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the shared feature surface used for training and future scoring."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = add_calendar_features(out, "Date")
    out = pd.concat(
        [compute_fwi_proxy(group) for _, group in out.groupby("City", sort=False)],
        ignore_index=True,
    )
    out = add_wildfire_weather_features(out)
    lag_vars = BASE_WEATHER + ["FWI_proxy", "VPD_kPa", "dry_days_streak"]
    out = _add_lag_roll_features(out, lag_vars)
    out = pd.get_dummies(out, columns=["City"], prefix="city", dtype=int)
    return out


def load_training_frame() -> pd.DataFrame:
    df = pd.read_parquet(ENG_DAILY)
    df["Date"] = pd.to_datetime(df["Date"])
    needed = ["City", "Date", TARGET] + BASE_WEATHER + STATIC_FEATURES
    existing = [c for c in needed if c in df.columns]
    train = df[existing].copy()
    for col in STATIC_FEATURES:
        if col not in train.columns:
            train[col] = 0.0
    return build_features(train)


def load_forecast_frame(history_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    forecast = pd.read_parquet(FORECAST_30D)
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    latest_static = (
        history_raw.sort_values("Date")
        .groupby("City", as_index=False)
        .tail(1)[["City"] + [c for c in STATIC_FEATURES if c in history_raw.columns]]
    )
    forecast = forecast.merge(latest_static, on="City", how="left")
    for col in STATIC_FEATURES:
        if col not in forecast.columns:
            forecast[col] = 0.0
    forecast[TARGET] = np.nan

    history_tail = history_raw[history_raw["Date"] >= forecast["Date"].min() - pd.Timedelta(days=45)].copy()
    combined = pd.concat([history_tail, forecast], ignore_index=True, sort=False)
    features = build_features(combined)
    future_features = features[features["Date"].isin(forecast["Date"])].copy()
    return forecast, future_features


def candidate_models(pos_weight: float) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=260, max_depth=16, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]),
        "ExtraTrees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=320, max_depth=18, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=31, learning_rate=0.055, l2_regularization=0.05, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=260, max_depth=5, learning_rate=0.045, subsample=0.85, colsample_bytree=0.85, eval_metric="aucpr", scale_pos_weight=pos_weight, random_state=RANDOM_SEED, n_jobs=-1)),
        ])
    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=360, max_depth=7, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, is_unbalance=True, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)),
        ])
    if CatBoostClassifier is not None:
        models["CatBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(iterations=280, depth=6, learning_rate=0.045, auto_class_weights="Balanced", eval_metric="AUC", random_seed=RANDOM_SEED, verbose=False)),
        ])
    return models


def feature_matrix(df: pd.DataFrame, feature_columns: List[str] | None = None) -> Tuple[pd.DataFrame, List[str]]:
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in DROP_COLUMNS if c in numeric.columns], errors="ignore")
    if feature_columns is None:
        feature_columns = sorted([c for c in numeric.columns if c != TARGET])
    return numeric.reindex(columns=feature_columns), feature_columns


def threshold_from_validation(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.81, 0.01):
        preds = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if recall < 0.45:
            continue
        score = 0.45 * average_precision_score(y_true, probabilities) + 0.35 * f1 + 0.20 * recall + 0.10 * precision
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(name: str, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
    val_prob_raw = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_prob_raw, y_val.to_numpy())
    val_prob = calibrator.predict(val_prob_raw)
    threshold = threshold_from_validation(y_val.to_numpy(), val_prob)
    test_prob = calibrator.predict(model.predict_proba(X_test)[:, 1])
    test_pred = (test_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average="binary", zero_division=0)
    metrics = {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "positive_rate": float(test_pred.mean()),
        "selection_score": float(0.45 * average_precision_score(y_test, test_prob) + 0.35 * f1 + 0.20 * recall + 0.10 * precision),
    }
    return ModelResult(name=name, estimator=model, calibrator=calibrator, threshold=threshold, metrics=metrics)


def train_and_select(features: pd.DataFrame) -> Tuple[ModelResult, List[Dict[str, float]], List[str]]:
    train_mask = features["Date"] < TRAIN_END
    val_mask = (features["Date"] >= TRAIN_END) & (features["Date"] < TEST_START)
    test_mask = features["Date"] >= TEST_START

    X_train, feature_columns = feature_matrix(features[train_mask])
    X_val, _ = feature_matrix(features[val_mask], feature_columns)
    X_test, _ = feature_matrix(features[test_mask], feature_columns)
    y_train = features.loc[train_mask, TARGET].astype(int)
    y_val = features.loc[val_mask, TARGET].astype(int)
    y_test = features.loc[test_mask, TARGET].astype(int)

    neg = max((y_train == 0).sum(), 1)
    pos = max((y_train == 1).sum(), 1)
    pos_weight = min(neg / pos, 20)

    results: List[ModelResult] = []
    for name, model in candidate_models(pos_weight).items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_val, y_val, X_test, y_test))

    results = sorted(results, key=lambda r: r.metrics["selection_score"], reverse=True)
    leaderboard = [{"model": r.name, **r.metrics} for r in results]
    return results[0], leaderboard, feature_columns


def write_outputs(best: ModelResult, leaderboard: List[Dict[str, float]], feature_columns: List[str], forecast_raw: pd.DataFrame, forecast_features: pd.DataFrame) -> None:
    X_future, _ = feature_matrix(forecast_features, feature_columns)
    probabilities = best.calibrator.predict(best.estimator.predict_proba(X_future)[:, 1])

    out = forecast_raw.copy().sort_values(["Date", "City"]).reset_index(drop=True)
    out["probability"] = probabilities
    out["confidence"] = out["probability"].map(_confidence)
    out["risk_level"] = out["probability"].map(_risk_level)
    out["predicted_fire"] = (out["probability"] >= best.threshold).astype(int)
    out["risk_score"] = (out["probability"] * 100).round(1)
    out["temperature"] = out["Temperature_C_mean"].round(1)
    out["wind"] = out["Wind_Speed_kmh_mean"].round(1)
    out["humidity"] = out["Humidity_percent_mean"].round(1)
    out["rain"] = out["Rain_mm_sum"].round(2)
    out["climate_summary"] = out.apply(_climate_summary, axis=1)
    out["warning"] = out.apply(_warning_text, axis=1)
    out["risk_color"] = out["risk_level"].map(RISK_COLORS)
    out["date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out["region"] = out["City"]

    public_cols = [
        "date",
        "region",
        "risk_level",
        "probability",
        "confidence",
        "risk_score",
        "predicted_fire",
        "temperature",
        "wind",
        "humidity",
        "rain",
        "Temperature_C_mean",
        "Humidity_percent_mean",
        "Rain_mm_sum",
        "Wind_Speed_kmh_mean",
        "Pressure_hPa_mean",
        "Solar_Radiation_Wm2_mean",
        "Soil_Temp_C_mean",
        "Soil_Moisture_mean",
        "Latitude",
        "Longitude",
        "climate_summary",
        "warning",
        "risk_color",
    ]
    out_public = out[public_cols].copy()
    out_public.to_csv(OUTPUTS / "forecast_30_days.csv", index=False)
    (OUTPUTS / "forecast_30_days.json").write_text(out_public.to_json(orient="records", indent=2), encoding="utf-8")

    latest = out_public.sort_values("date").groupby("region", as_index=False).tail(1)
    map_points = latest.to_dict(orient="records")
    (OUTPUTS / "map_points.json").write_text(json.dumps(map_points, indent=2), encoding="utf-8")

    metrics = {
        "generated_at": pd.Timestamp.now(tz="Asia/Baku").isoformat(),
        "prediction_horizon_days": 30,
        "target": "Daily probability of a NASA FIRMS wildfire detection within the city risk area",
        "selected_model": best.name,
        "selected_threshold": best.threshold,
        "temporal_split": {
            "train": f"< {TRAIN_END.date()}",
            "validation": f"{TRAIN_END.date()} to {TEST_START.date()}",
            "test": f">= {TEST_START.date()}",
        },
        "leaderboard": leaderboard,
        "feature_count": len(feature_columns),
        "risk_levels": {
            "Low": "< 15%",
            "Moderate": "15% to 35%",
            "High": "35% to 60%",
            "Extreme": ">= 60%",
        },
        "data_sources": ["NASA FIRMS MODIS/VIIRS", "Open-Meteo ERA5/ERA5-Land", "Open-Elevation/static geography"],
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    MODELS_F.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best.estimator, "calibrator": best.calibrator, "threshold": best.threshold, "features": feature_columns},
        MODELS_F / "forecast_compatible_fire_model.joblib",
    )


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS_F.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ENG_DAILY)
    raw["Date"] = pd.to_datetime(raw["Date"])
    train_features = load_training_frame()
    best, leaderboard, feature_columns = train_and_select(train_features)
    forecast_raw, forecast_features = load_forecast_frame(raw)
    write_outputs(best, leaderboard, feature_columns, forecast_raw, forecast_features)
    print(f"Selected model: {best.name}")
    print(f"Outputs written to: {OUTPUTS}")


if __name__ == "__main__":
    main()
