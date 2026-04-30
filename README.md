# ARIAN Wildfire Intelligence

ARIAN is a 30-day wildfire risk prediction and dashboard project for Azerbaijan. It combines NASA FIRMS satellite fire detections, Open-Meteo weather history and forecasts, static geography, and engineered fire-weather features to produce readable regional wildfire risk forecasts.

## What This Project Does

- Trains multiple wildfire classifiers with a temporal split to avoid data leakage.
- Uses forecast-compatible weather, climate, and geography features.
- Calibrates probabilities on validation data so risk levels are meaningful.
- Generates dashboard-ready forecast outputs.
- Provides a professional web dashboard with an Azerbaijan map, region panel, risk calendar, charts, and filters.

## Data Sources

- NASA FIRMS MODIS and VIIRS fire detections
- Open-Meteo ERA5 / ERA5-Land weather history and forecasts
- Open-Elevation and static geography inputs
- Local legacy geography, vegetation, roads, and population files where available

## Project Structure

```text
ARIAN/
  data/
    raw/                 # FIRMS, weather, and legacy source data
    processed/           # master and engineered parquet datasets
    reference/           # city coordinates and static geography
  dashboard/
    index.html           # interactive web dashboard
    app.js               # dashboard interactions and charts
    styles.css           # professional blue/white UI
  models/
    wildfire/            # trained wildfire model artifacts
  notebooks/             # original research notebooks
  outputs/
    forecast_30_days.csv
    forecast_30_days.json
    map_points.json
    metrics.json
  reports/               # figures, maps, and model metrics
  src/
    config.py
    features.py
    modeling.py
    evaluation.py
    prediction_pipeline.py
```

## Run the Prediction Pipeline

Install dependencies:

```powershell
pip install -r requirements.txt
```

Regenerate the 30-day forecast outputs:

```powershell
python -m src.prediction_pipeline
```

This writes:

- `outputs/forecast_30_days.csv`
- `outputs/forecast_30_days.json`
- `outputs/map_points.json`
- `outputs/metrics.json`

## Run the Dashboard

Start a local static server from the project root:

```powershell
python -m http.server 8000
```

Open:

```text
http://localhost:8000/dashboard/
```

The dashboard reads the generated JSON files from `outputs/`.

## Modeling Approach

The current production pipeline predicts daily wildfire risk for each region/city. The target is whether a NASA FIRMS fire detection occurs for that city-day risk area.

The pipeline:

1. Loads `data/processed/engineered_daily.parquet`.
2. Builds a forecast-compatible feature set from weather, lagged weather, rolling weather, FWI proxies, VPD, date signals, city indicators, and geography.
3. Splits data by time:
   - Train: before `2024-01-01`
   - Validation: `2024-01-01` to `2025-01-01`
   - Test: `2025-01-01` onward
4. Compares Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, and CatBoost when installed.
5. Calibrates probabilities with validation data.
6. Selects the best model using PR-AUC, F1, recall, and precision.
7. Scores the 30-day weather forecast.

## Dashboard Features

- Hero section and project intention
- Azerbaijan wildfire risk map
- Normal and satellite map layers
- Clickable/hoverable city risk markers
- Dynamic region panel with risk, temperature, wind, humidity, confidence, climate summary, and warning text
- 30-day color-coded risk calendar
- Region-specific trend and weather charts
- Filterable forecast table
- Professional footer with mission and data sources

## Notes

The dashboard intentionally keeps model metrics in the background. Technical metrics are stored in `outputs/metrics.json`, while the UI focuses on clear environmental risk communication.
