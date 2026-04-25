"""
FastAPI backend for the Azerbaijan weather + wildfire dashboard.

Serves pre-computed artefacts from Phases 2-4 via three endpoint families:

    /weather/*    -- 30-day weather forecast
    /wildfire/*   -- 30-day wildfire risk
    /insights/*   -- climate trends + headline answers + forecast anomalies

Run from the project root::

    uvicorn app.api.main:app --reload --port 8000

Design notes
------------
* **Data is loaded lazily on startup**, cached in a lightweight service
  singleton, and refreshed on demand via ``POST /admin/refresh``. A proper
  production build would put this behind an async task queue (Celery,
  Dramatiq) and treat refreshes as scheduled jobs.
* **Pydantic response models** give automatic OpenAPI schema + client codegen.
* **CORS is wide open** for development; tighten ``allowed_origins`` for prod.
* **Root is mounted on the static dashboard** so the whole thing is a single
  service; in production split these into CDN + API gateway.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import List, Optional

# --- Make project root importable when served via uvicorn -----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.api.service import DataService

logger = logging.getLogger("wildfire_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ----------------------------------------------------------------------------
# Lifespan — load all artefacts once at startup
# ----------------------------------------------------------------------------

service: Optional[DataService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and tear down the data-serving singleton."""
    global service
    logger.info("Starting DataService ...")
    service = DataService()
    service.load_all()
    logger.info("DataService ready: %s", service.summary())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Azerbaijan Wildfire Forecasting API",
    description=(
        "30-day weather forecast, climate analysis, and wildfire risk "
        "predictions for 5 Azerbaijani cities."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Response models
# ----------------------------------------------------------------------------

class WeatherPoint(BaseModel):
    city: str = Field(..., description="City name")
    forecast_date: date = Field(..., description="Forecast target date")
    horizon_days: int = Field(..., ge=1, le=30)
    target: str
    y_pred: float


class WeatherDayWide(BaseModel):
    city: str
    forecast_date: date
    horizon_days: int
    temperature_2m: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    wind_direction_10m: Optional[float] = None
    rain: Optional[float] = None
    precipitation: Optional[float] = None


class WildfireDay(BaseModel):
    city: str
    forecast_date: date
    horizon_days: int
    fire_probability: float = Field(..., ge=0, le=1)
    risk_category: str
    expected_fire_count: float = Field(..., ge=0)


class InsightSummary(BaseModel):
    city: str
    next_30d_temperature: str
    next_30d_rainfall: str
    wind_trend: str
    mean_fire_probability: float
    headline: str


class MetaInfo(BaseModel):
    cities: List[str]
    anchor_date: date
    forecast_start: date
    forecast_end: date
    weather_targets: List[str]
    generated_at: str


# ----------------------------------------------------------------------------
# Health + metadata
# ----------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    if service is None or not service.is_loaded:
        raise HTTPException(503, "Service not initialised")
    return {"status": "ok", "artefacts_loaded": service.is_loaded}


@app.get("/meta", response_model=MetaInfo, tags=["meta"])
def meta():
    if service is None: raise HTTPException(503, "Service not ready")
    return service.meta()


# ----------------------------------------------------------------------------
# Weather endpoints
# ----------------------------------------------------------------------------

@app.get("/weather", response_model=List[WeatherDayWide], tags=["weather"])
def weather_wide(
    city: Optional[str] = Query(None, description="City filter; omit for all cities"),
    horizon_max: int = Query(30, ge=1, le=30, description="Max horizon to return"),
):
    """Wide-format forecast: one row per (city, date) with every target as
    its own column. Ideal for plotting."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.weather_wide(city=city, horizon_max=horizon_max)


@app.get("/weather/long", response_model=List[WeatherPoint], tags=["weather"])
def weather_long(
    city: Optional[str] = None,
    target: Optional[str] = Query(None, description="e.g. temperature_2m, rain"),
):
    """Long-format forecast: one row per (city, date, target)."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.weather_long(city=city, target=target)


# ----------------------------------------------------------------------------
# Wildfire endpoints
# ----------------------------------------------------------------------------

@app.get("/wildfire", response_model=List[WildfireDay], tags=["wildfire"])
def wildfire(
    city: Optional[str] = None,
    risk_min: Optional[str] = Query(None, description="Filter: low/moderate/high/very_high"),
):
    """30-day wildfire risk per city per day."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.wildfire(city=city, risk_min=risk_min)


@app.get("/wildfire/summary", tags=["wildfire"])
def wildfire_summary():
    """Per-city rollup of fire probability over the 30-day window."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.wildfire_summary()


# ----------------------------------------------------------------------------
# Insights endpoints
# ----------------------------------------------------------------------------

@app.get("/insights", response_model=List[InsightSummary], tags=["insights"])
def insights():
    """One row per city combining every headline finding."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.insights()


@app.get("/insights/trends", tags=["insights"])
def trends(city: Optional[str] = None, variable: Optional[str] = None):
    """Annual Mann-Kendall + Theil-Sen trend per (city, variable)."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.trends(city=city, variable=variable)


@app.get("/insights/anomalies", tags=["insights"])
def anomalies(
    city: Optional[str] = None,
    target: Optional[str] = Query(None, description="e.g. temperature_2m, precipitation"),
):
    """Forecast-vs-climatology z-scores per city per day per target."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.anomalies(city=city, target=target)


@app.get("/insights/climatology", tags=["insights"])
def climatology(city: str, variable: str):
    """Daily DOY climatology for a given city and variable (mean + std)."""
    if service is None: raise HTTPException(503, "Service not ready")
    return service.climatology(city=city, variable=variable)


# ----------------------------------------------------------------------------
# Admin
# ----------------------------------------------------------------------------

@app.post("/admin/refresh", tags=["admin"])
def refresh():
    """Reload all CSV artefacts from disk (useful after a nightly pipeline run)."""
    if service is None: raise HTTPException(503, "Service not ready")
    service.load_all()
    return {"status": "reloaded", "summary": service.summary()}


# ----------------------------------------------------------------------------
# Static dashboard at /
# ----------------------------------------------------------------------------

STATIC_DIR = PROJECT_ROOT / "app" / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(STATIC_DIR)), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(STATIC_DIR / "index.html"))
