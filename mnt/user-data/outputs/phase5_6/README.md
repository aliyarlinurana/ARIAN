# Azerbaijan Weather → Climate → Wildfire Forecasting System

A production-grade data pipeline that (1) forecasts daily weather for the next
30 days, (2) compares those forecasts against 6 years of historical
climatology, and (3) predicts wildfire occurrence risk per city per day.

Built with a **strict layered architecture**: Weather is the core engine;
Climate is a statistical comparison layer on top; Wildfire depends on both.
No layer is allowed to leak information backward or forward.

---

## Project at a glance

| Metric | Value |
|---|---|
| Study area | Azerbaijan — 5 cities with hourly weather (Baku, Ganja, Guba, Lankaran, Zaqatala); 16 with auxiliary data |
| History window | 2020-01-01 → 2026-04-18 (daily) |
| Forecast horizon | 30 days |
| Weather forecast RMSE (h=1, temperature) | **1.57 °C** (vs persistence 1.73 °C) |
| Wildfire classifier ROC-AUC (2024 holdout) | **0.79** |
| Wildfire regressor MAE (2024 holdout) | **0.27** (vs mean 0.48) |
| Pipeline runtime (end-to-end, single CPU) | ~6-8 minutes |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Data Ingestion                                        │
│  Open-Meteo (hourly), WorldPop, OSM roads, MODIS NDVI,           │
│  NASA FIRMS, lightning climatology, forest KMZ                   │
│  →  data/interim/*.csv                                           │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Weather Forecasting System (CORE)                     │
│    2.1 Cleaning: align grid, clip impossibilities, circular-     │
│        aware daily wind, outlier flagging (non-destructive)      │
│    2.2 Feature engineering: lags, rolling stats, Fourier, wind   │
│        vectors, calendar (495 features total)                    │
│    2.3 Modeling: persistence, ridge, HistGradientBoosting;       │
│        time-ordered CV, leak-free horizon shift                  │
│    2.4 Forecasting: 5 target × 5 anchor horizons = 25 models,    │
│        linear interpolation between anchors → 30-day dataset     │
│  →  data/processed/weather_forecast.csv                          │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Climate Analysis                                      │
│    Daily + monthly climatologies (7-day smoothed DOY mean/std)   │
│    Robust trend tests (Mann-Kendall + Theil-Sen)                 │
│    Forecast vs climatology anomalies + z-scores                  │
│  →  reports/climate/*.csv                                        │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 4 — Wildfire System                                       │
│    4.1 Feature engineering: drought index, heatwave indicator,   │
│        wind-spread factor, NDVI anomaly, human-activity score,   │
│        lightning climatology. Target = FIRMS vegetation hotspots │
│        within 50 km of each city (type=0, confidence ≠ low)      │
│    4.2 Modeling: classifier (fire_occurred) + regressor          │
│        (fire_count). Logistic + HGBC + calibrated probabilities  │
│    4.3 Prediction: forecast + history tail + climatology fill    │
│        → 30-day risk dataset per city                            │
│  →  data/processed/wildfire_risk_forecast.csv                    │
└──────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 5 — Final Analysis & Dashboard                            │
│    Executive report answering 5 project questions                │
│    FastAPI backend + dashboard frontend                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Repository layout

```
project_root/
├── data/
│   ├── raw/
│   │   ├── earth_engine/           # MODIS NDVI, Copernicus landcover
│   │   ├── forest_boundaries/      # KMZ polygons
│   │   ├── lightning/              # Monthly thunder-hours grids
│   │   ├── nasa_firms/             # VIIRS fire hotspots
│   │   ├── openmeteo/              # Hourly + daily weather
│   │   ├── osm_roads/              # Road-density feature per city
│   │   └── population/             # WorldPop by year
│   ├── interim/                    # Phase 1+2.1 outputs
│   └── processed/                  # Feature matrices + forecasts
│
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_weather_cleaning.ipynb
│   ├── 03_weather_feature_engineering.ipynb
│   ├── 04_weather_modeling.ipynb
│   ├── 05_weather_forecasting.ipynb
│   ├── 06_climate_analysis.ipynb
│   ├── 07_wildfire_feature_engineering.ipynb
│   ├── 08_wildfire_modeling.ipynb
│   ├── 09_wildfire_prediction.ipynb
│   └── 10_final_analysis.ipynb
│
├── src/
│   ├── weather/
│   │   ├── ingestion.py            # Open-Meteo + auxiliary loaders
│   │   ├── cleaning.py             # Gap alignment, clipping, circular-aware
│   │   ├── features.py             # Leak-safe temporal features
│   │   ├── train.py                # Multi-model comparison
│   │   └── forecast.py             # 30-day multi-horizon
│   ├── climate/
│   │   └── trends.py               # Mann-Kendall, Theil-Sen, anomalies
│   ├── wildfire/
│   │   ├── features.py             # Fire-science feature engineering
│   │   ├── train.py                # Classifier + regressor
│   │   └── predict.py              # 30-day risk assembly
│   └── utils/
│       ├── config.py               # Single source of truth: paths, cities, targets
│       └── logging_utils.py
│
├── models/
│   ├── weather/                    # 30 joblib models (5 targets × 5 horizons + h=1 set)
│   └── wildfire/                   # 6 joblib models (classifier + regressor)
│
├── reports/
│   └── climate/                    # Climatology CSVs + headline answers
│
├── app/                            # FastAPI backend + dashboard frontend
└── README.md
```

---

## Data sources

| Source | Description | Cadence | Coverage |
|---|---|---|---|
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | Hourly weather variables (24 params) | Hourly | 2020-01-01 → 2026-04-18 |
| [NASA FIRMS VIIRS](https://firms.modaps.eosdis.nasa.gov/) | Active fire hotspots | ~daily | 2020-2024 |
| [MODIS NDVI (GEE)](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1) | 16-day vegetation index | 16-day | 2020-2026 |
| [OpenStreetMap](https://www.openstreetmap.org/) | Road network for human-access | Static | 2024 snapshot |
| [WorldPop](https://www.worldpop.org/) | Gridded population density | Annual | 2020-2026 |
| [Lightning climatology](https://ltg.meteo.com/) | Monthly thunder-hours | Monthly | 2020-2024 |

---

## Setup

### Requirements

- Python 3.10+
- ~500 MB disk space for data + models
- RAM: 4 GB is enough for a full end-to-end run on one CPU

### Install

```bash
git clone <your-repo-url>
cd wildfire_project
pip install -r requirements.txt
```

`requirements.txt` core libraries:

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
scipy>=1.11
joblib>=1.3
matplotlib>=3.7
pyarrow>=14.0          # for parquet persistence (optional)
```

Optional (activate additional model paths):

```
xgboost>=2.0           # unlocks XGBoost classifier/regressor
statsmodels>=0.14      # unlocks SARIMA per-city baseline
prophet>=1.1           # unlocks Prophet
geopandas>=0.14        # unlocks forest KMZ + raster landcover
rasterio>=1.3
openmeteo-requests     # if re-pulling weather from the API
```

### Environment variables (optional)

None strictly required. If re-fetching weather from the API you may want
a persistent `requests_cache` directory — the script places it in `.cache/`
by default.

---

## End-to-end usage

### Run the whole pipeline from scratch

```bash
# Phase 1 — ingestion
python -m src.weather.ingestion

# Phase 2 — weather
python -m src.weather.cleaning
python -m src.weather.features
python -m src.weather.train
python -m src.weather.forecast

# Phase 3 — climate
python -m src.climate.trends

# Phase 4 — wildfire
python -m src.wildfire.features
python -m src.wildfire.train
python -m src.wildfire.predict
```

Or execute the notebooks in order (01 → 10).

### Programmatic use

```python
from src.weather.forecast import make_30day_forecast
from src.wildfire.predict import predict_wildfire_risk_30day

# 30-day weather forecast (750 rows)
forecast = make_30day_forecast(algo="hgbr")

# 30-day wildfire risk (150 rows: 5 cities × 30 days)
risk = predict_wildfire_risk_30day()
```

---

## Results (anchor date 2026-04-18)

### Weather forecast quality (holdout 2025 → h=1)

| Target | Metric | Value | Persistence baseline |
|---|---|---|---|
| temperature_2m | RMSE | 1.57 °C | 1.73 °C |
| wind_speed_10m | RMSE | 3.39 m/s | 3.87 m/s |
| wind_direction_10m | circ RMSE | 93° | — |
| rain | RMSE | 5.82 mm | 7.20 mm |
| precipitation | RMSE | 5.99 mm | 7.38 mm |

Forecast skill decays with horizon roughly as expected from operational
meteorology: useful temperature skill to ~14 days, precipitation to ~5 days.

### Climate headlines (2020-2025)

- **No statistically significant annual-temperature trend** in any city
  (Mann-Kendall p > 0.05). Six years is too short to separate trend from
  interannual variability.
- **Rainfall trend not detectable**, though Zaqatala is marginally increasing
  (p = 0.06, slope ≈ +79 mm/yr).
- **Wind speed decreasing** in Guba and Lankaran at p < 0.05 (Theil-Sen
  slope ≈ −0.1 to −0.23 m/s/yr).

### Wildfire risk (next 30 days, anchored 2026-04-18)

| City | Mean P(fire) | Verdict |
|---|---|---|
| Baku | 0.57 | VERY HIGH |
| Ganja | 0.29 | HIGH |
| Guba | 0.19 | MODERATE |
| Zaqatala | 0.14 | LOW |
| Lankaran | 0.13 | LOW |

---

## Design decisions worth reading

These choices are deliberate, not accidental. If you're extending the pipeline,
know why they're here before changing them:

1. **Target shift for forecasting.** Weather targets are shifted forward by
   the forecast horizon BEFORE training. This prevents same-day feature
   leakage (e.g. `temperature_max` at time `t` trivially reconstructing
   `temperature_mean` at `t`).
2. **Leak-safe rolling features.** Every rolling statistic is
   `series.shift(1).rolling(w).agg()`, so the row at `t` sees only data
   from `t-w..t-1`.
3. **Circular-aware wind direction.** Daily wind direction is built from
   speed-weighted unit vectors, not scalar averaging. Metrics use
   minimum-arc distance.
4. **Non-destructive outlier flagging.** Outliers are flagged, not removed.
   Models can still see the raw distribution; FE can choose what to down-weight.
5. **Sparse horizon ladder + linear interpolation.** We train 5 horizons
   (1, 3, 7, 14, 30) per target and interpolate between, instead of 30
   separate models. Same total dynamic range, 6× less compute.
6. **FIRMS filtering.** Only `type=0` (vegetation) hotspots with
   confidence ≠ "l" are counted. This removes most industrial flaring
   (especially in Baku) from the fire target.
7. **50 km city-radius for fire attribution.** Chosen to give healthy
   positive rates (9-27%) and to keep the problem tractable for pooled
   models. Smaller radii give sparse targets; larger ones overlap between
   cities.
8. **Pooled cross-city models with city dummies.** One model per target
   (not one per city) for data efficiency. City dummies let the model
   specialise per location where it needs to.
9. **Calibrated classifier probabilities.** HGBC output is wrapped in
   `CalibratedClassifierCV` with isotonic calibration so `predict_proba`
   returns a trustworthy risk number, not a raw score.
10. **Rebuild forecast-time features, don't reuse training features.** The
    wildfire prediction pipeline rebuilds the 35-feature matrix from the
    weather forecast using the same feature functions the training pipeline
    used — preventing "works in training, fails in production" bugs.

---

## Known limitations (honest version)

1. **5 years ≠ climatology.** WMO recommends 30 years. Our trend tests are
   underpowered; "no trend detected" does not mean "no trend exists".
2. **5 cities of hourly weather, 16 of auxiliaries.** Extending weather
   ingestion to all 16 cities enables full-country coverage.
3. **FIRMS label noise.** Some industrial hotspots leak through the `type=0`
   filter, particularly in Baku. A dedicated gas-flare mask would sharpen
   the Baku risk estimate.
4. **Weather skill decays with horizon.** Beyond day 7-14, forecasts
   converge to climatology. Long-horizon wildfire risk reflects seasonal
   expectation more than dynamic prediction.
5. **No leave-one-city-out evaluation.** Pooled models were not tested on
   held-out cities. Deploying to new cities should include LOCO validation.

---

## Future improvements

1. **Extend weather history** with ERA5 reanalysis (30+ years) for trend
   tests with real statistical power.
2. **Gas-flare mask** for Azerbaijan to clean up FIRMS labels in Baku.
3. **Ensemble weather forecasting** (Prophet + SARIMA + HGBR blend) for
   calibrated uncertainty intervals.
4. **Probabilistic wildfire output** — Bayesian or quantile regression so
   the dashboard can show "P90 expected fires" bands.
5. **Daily refresh pipeline** — cron job that re-pulls Open-Meteo, rebuilds
   features, and refreshes predictions without re-training.
6. **Hierarchical forecasting** so daily, weekly, and monthly sums are
   reconciled (avoids "30 daily predictions sum to something different from
   the monthly prediction").
7. **SHAP values** on the wildfire classifier for per-prediction
   explanations in the dashboard.

---

## License

MIT.

## Citation

If you use this pipeline or any of its outputs in academic work, please
cite the data sources (Open-Meteo, NASA FIRMS, MODIS, WorldPop, OSM,
Blitzortung lightning) separately. This pipeline is a novel integration
but depends entirely on their open data.

---

## Quick links

- **Full report**: `notebooks/10_final_analysis.ipynb`
- **Weather forecast CSV**: `data/processed/weather_forecast.csv`
- **Wildfire risk CSV**: `data/processed/wildfire_risk_forecast.csv`
- **Climate headlines**: `reports/climate/headline_answers.csv`
- **Dashboard**: `app/` (after Phase 7)
