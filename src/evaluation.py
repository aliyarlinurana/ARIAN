"""
ARIAN Wildfire Prediction — Evaluation Utilities
==================================================
Metrics, threshold tuning, leaderboard helpers.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)


# ═══════════════════════════════════════════════════════════════════════════
# Classification (Fire Detection)
# ═══════════════════════════════════════════════════════════════════════════

def fire_metrics(y_true, y_pred, y_prob=None):
    """Compute comprehensive fire detection metrics."""
    m = {
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
        "fn":        int(((y_true == 1) & (y_pred == 0)).sum()),
        "fp":        int(((y_true == 0) & (y_pred == 1)).sum()),
        "tp":        int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn":        int(((y_true == 0) & (y_pred == 0)).sum()),
    }
    if y_prob is not None:
        try:
            m["pr_auc"]  = average_precision_score(y_true, y_prob)
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            m["pr_auc"]  = 0.0
            m["roc_auc"] = 0.0
    return m


def find_optimal_threshold(y_true, y_prob,
                           recall_weight=0.3, f1_weight=0.4,
                           precision_weight=0.3,
                           min_precision=0.30, min_recall=0.60,
                           grid=None):
    """Find threshold that maximises weighted recall+precision+f1 subject to
    min precision (>=0.30) and min recall (>=0.60) hard gates.
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.01)

    best_thresh, best_score = 0.5, -1
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        # Reject thresholds that break hard gates
        if prec < min_precision or rec < min_recall:
            continue

        score = recall_weight * rec + precision_weight * prec + f1_weight * f1
        if score > best_score:
            best_score  = score
            best_thresh = t

    return best_thresh


def build_fire_leaderboard(results_dict):
    """Build a leaderboard DataFrame from results dictionary.

    results_dict: {model_name: {"y_pred": ..., "y_prob": ..., "threshold": ...,
                                "imbalance_strategy": ..., ...}}
    """
    rows = []
    for name, r in results_dict.items():
        m = fire_metrics(r["y_true"], r["y_pred"], r.get("y_prob"))
        m["model"] = name
        m["threshold"] = r.get("threshold", 0.5)
        m["imbalance_strategy"] = r.get("imbalance_strategy", "none")
        rows.append(m)

    if not rows:
        return pd.DataFrame(columns=[
            "model","threshold","imbalance_strategy",
            "recall","precision","f1","accuracy",
            "fn","fp","tp","tn","pr_auc","roc_auc","composite",
        ])

    lb = pd.DataFrame(rows)
    # Sort by weighted recall+f1 composite score
    for col in ("recall", "precision", "f1"):
        if col not in lb.columns:
            lb[col] = 0.0
    lb["composite"] = 0.3 * lb["recall"] + 0.3 * lb["precision"] + 0.4 * lb["f1"]
    lb = lb.sort_values("composite", ascending=False).reset_index(drop=True)
    return lb


# ═══════════════════════════════════════════════════════════════════════════
# Regression (Weather)
# ═══════════════════════════════════════════════════════════════════════════

def weather_metrics(y_true, y_pred, var_name=""):
    """Compute regression metrics."""
    y_t = np.array(y_true).ravel()
    y_p = np.array(y_pred).ravel()
    mask = np.isfinite(y_t) & np.isfinite(y_p)
    y_t, y_p = y_t[mask], y_p[mask]

    mae  = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2   = r2_score(y_t, y_p) if len(y_t) > 1 else 0.0

    # MAPE — only where y_true != 0
    nz = y_t != 0
    mape = np.mean(np.abs((y_t[nz] - y_p[nz]) / y_t[nz])) * 100 if nz.any() else np.nan

    return {"variable": var_name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
