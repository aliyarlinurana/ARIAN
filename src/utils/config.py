"""
Project-wide configuration constants.

Single source of truth for:
    - Filesystem layout (raw / interim / processed / models / reports)
    - Study-area metadata (cities and their coordinates)
    - Weather API variable lists

Importing from this module keeps notebook code and production scripts in sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ----------------------------------------------------------------------------
# Filesystem layout
# ----------------------------------------------------------------------------
# Resolve project root = two levels up from this file: src/utils/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# Sub-directories per data source under data/raw/
RAW_OPENMETEO: Path = RAW_DIR / "openmeteo"
RAW_POPULATION: Path = RAW_DIR / "population"
RAW_OSM_ROADS: Path = RAW_DIR / "osm_roads"
RAW_EARTH_ENGINE: Path = RAW_DIR / "earth_engine"
RAW_NASA_FIRMS: Path = RAW_DIR / "nasa_firms"
RAW_LIGHTNING: Path = RAW_DIR / "lightning"
RAW_FOREST_BOUNDARIES: Path = RAW_DIR / "forest_boundaries"


# ----------------------------------------------------------------------------
# Study area: Azerbaijan cities
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CityPoint:
    """Immutable geographic anchor for a study-area city."""

    name: str
    lat: float
    lon: float


AZERBAIJAN_CITIES: Dict[str, CityPoint] = {
    "Baku":        CityPoint("Baku",        40.4093, 49.8671),
    "Ganja":       CityPoint("Ganja",       40.6828, 46.3606),
    "Lankaran":    CityPoint("Lankaran",    38.7529, 48.8515),
    "Guba":        CityPoint("Guba",        41.3597, 48.5134),
    "Zaqatala":    CityPoint("Zaqatala",    41.6336, 46.6433),
    "Nakhchivan":  CityPoint("Nakhchivan",  39.2089, 45.4122),
    "Sheki":       CityPoint("Sheki",       41.1919, 47.1706),
    "Shirvan":     CityPoint("Shirvan",     39.9317, 48.9290),
    "Mingachevir": CityPoint("Mingachevir", 40.7639, 47.0595),
    "Khachmaz":    CityPoint("Khachmaz",    41.4635, 48.8060),
    "Goychay":     CityPoint("Goychay",     40.6533, 47.7401),
    "Shamkir":     CityPoint("Shamkir",     40.8298, 46.0162),
    "Sabirabad":   CityPoint("Sabirabad",   40.0101, 48.4772),
    "Imishli":     CityPoint("Imishli",     39.8694, 48.0600),
    "Shamakhi":    CityPoint("Shamakhi",    40.6303, 48.6414),
    "Jalilabad":   CityPoint("Jalilabad",   39.2096, 48.4919),
}


def cities_as_records() -> List[Dict[str, float]]:
    """Return the city dict as a list of records (for DataFrame construction)."""
    return [
        {"City": c.name, "Latitude": c.lat, "Longitude": c.lon}
        for c in AZERBAIJAN_CITIES.values()
    ]


# ----------------------------------------------------------------------------
# Weather API configuration (Open-Meteo historical archive)
# ----------------------------------------------------------------------------
OPENMETEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_TIMEZONE: str = "Europe/Moscow"
DEFAULT_START_DATE: str = "2020-01-01"
DEFAULT_END_DATE: str = "2026-04-18"

HOURLY_WEATHER_VARS: List[str] = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "rain", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "vapour_pressure_deficit", "evapotranspiration", "wind_speed_10m",
    "wind_direction_10m", "wind_speed_80m", "wind_direction_80m", "temperature_80m",
    "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_temperature_28_to_100cm", "is_day", "sunshine_duration",
    "shortwave_radiation", "direct_radiation",
]

DAILY_WEATHER_VARS: List[str] = ["rain_sum"]

# Forecast targets for the weather system (Phase 2)
FORECAST_TARGETS: List[str] = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",  # circular — use sin/cos decomposition
    "rain",
    "precipitation",
]

# Projected CRS used for meter-based buffers over Azerbaijan (UTM zone 39N)
METRIC_CRS_EPSG: int = 32639
GEOGRAPHIC_CRS_EPSG: int = 4326

# Radius around city centroids used for human-access / road-density features
CITY_BUFFER_METERS: int = 5_000
