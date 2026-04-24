"""
Weather + auxiliary-source data ingestion.

Responsibilities
----------------
1. Fetch historical weather from the Open-Meteo archive for the study-area cities
   (or load a previously-downloaded snapshot from disk when the API is offline).
2. Load every auxiliary raw source (population, roads, NDVI, NASA FIRMS, lightning,
   forest boundaries, landcover) into tidy pandas / geopandas frames.
3. Validate the shape and coverage of each frame.
4. Persist standardised, analysis-ready copies to ``data/interim/``.

Design principles
-----------------
* **One function per source.** Each returns a DataFrame with documented columns.
* **Lazy heavy imports.** geopandas / rasterio / osmium are only imported inside
  the functions that need them so the core pipeline is usable even if those
  GDAL-dependent libraries are unavailable.
* **Fail loud, not silent.** Every loader validates its output and logs clear
  diagnostics. Missing-file errors name the file.
* **Pure, deterministic, side-effect free** for loaders; only ``save_interim``
  and the top-level orchestrator write to disk.

Public entry point: :func:`run_ingestion_pipeline`.
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Make "src" importable both as a package and in notebook sys.path setups
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    AZERBAIJAN_CITIES,
    CityPoint,
    DAILY_WEATHER_VARS,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    HOURLY_WEATHER_VARS,
    INTERIM_DIR,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_TIMEZONE,
    RAW_EARTH_ENGINE,
    RAW_FOREST_BOUNDARIES,
    RAW_LIGHTNING,
    RAW_NASA_FIRMS,
    RAW_OPENMETEO,
    RAW_OSM_ROADS,
    RAW_POPULATION,
    cities_as_records,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ============================================================================
# 1. WEATHER — Open-Meteo historical archive
# ============================================================================

def fetch_weather_from_api(
    cities: Optional[Dict[str, CityPoint]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    hourly_vars: Optional[List[str]] = None,
    daily_vars: Optional[List[str]] = None,
    polite_sleep_sec: float = 5.0,
    rate_limit_backoff_sec: float = 60.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download historical weather from Open-Meteo for every study-area city.

    Parameters
    ----------
    cities
        Mapping from name -> :class:`CityPoint`. Defaults to AZERBAIJAN_CITIES.
    start_date, end_date
        ``YYYY-MM-DD`` strings covering the requested archive window.
    hourly_vars, daily_vars
        Variable lists (see :mod:`src.utils.config` for defaults).
    polite_sleep_sec
        Pause between city requests to stay under the free-tier rate limit.
    rate_limit_backoff_sec
        Pause length when the server reports the minutely quota is exhausted.

    Returns
    -------
    (hourly_df, daily_df)
        Two tidy long-format DataFrames with a leading ``City`` column.

    Notes
    -----
    Requires the ``openmeteo_requests``, ``requests_cache`` and ``retry_requests``
    packages. If they are not installed a :class:`RuntimeError` is raised so the
    caller can fall back to :func:`load_weather_from_disk`.
    """
    try:
        import openmeteo_requests  # type: ignore
        import requests_cache  # type: ignore
        from retry_requests import retry  # type: ignore
    except ImportError as exc:  # pragma: no cover - network not available in CI
        raise RuntimeError(
            "openmeteo_requests / requests_cache / retry_requests are required for "
            "fetch_weather_from_api(). Install them or call load_weather_from_disk()."
        ) from exc

    cities = cities or AZERBAIJAN_CITIES
    hourly_vars = hourly_vars or HOURLY_WEATHER_VARS
    daily_vars = daily_vars or DAILY_WEATHER_VARS

    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)

    all_hourly, all_daily = [], []

    for name, pt in cities.items():
        logger.info("Fetching weather for %s (%.4f, %.4f)", name, pt.lat, pt.lon)

        params = {
            "latitude": pt.lat,
            "longitude": pt.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": hourly_vars,
            "daily": daily_vars,
            "models": "best_match",
            "timezone": OPENMETEO_TIMEZONE,
        }

        # --- retry loop for Open-Meteo minutely rate limit ---
        while True:
            try:
                response = client.weather_api(OPENMETEO_ARCHIVE_URL, params=params)[0]
                hourly = response.Hourly()
                hourly_block = {
                    "date": pd.date_range(
                        start=pd.to_datetime(
                            hourly.Time() + response.UtcOffsetSeconds(), unit="s", utc=True
                        ),
                        end=pd.to_datetime(
                            hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True
                        ),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left",
                    )
                }
                for i, v in enumerate(hourly_vars):
                    hourly_block[v] = hourly.Variables(i).ValuesAsNumpy()

                daily = response.Daily()
                daily_block = {
                    "date": pd.date_range(
                        start=pd.to_datetime(
                            daily.Time() + response.UtcOffsetSeconds(), unit="s", utc=True
                        ),
                        end=pd.to_datetime(
                            daily.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True
                        ),
                        freq=pd.Timedelta(seconds=daily.Interval()),
                        inclusive="left",
                    )
                }
                for i, v in enumerate(daily_vars):
                    daily_block[v] = daily.Variables(i).ValuesAsNumpy()

                city_hourly = pd.DataFrame(hourly_block); city_hourly.insert(0, "City", name)
                city_daily = pd.DataFrame(daily_block);   city_daily.insert(0, "City", name)
                all_hourly.append(city_hourly)
                all_daily.append(city_daily)
                logger.info("  OK %s: %d hourly rows, %d daily rows",
                            name, len(city_hourly), len(city_daily))
                time.sleep(polite_sleep_sec)
                break
            except Exception as exc:  # noqa: BLE001 — we inspect the message
                msg = str(exc)
                if "Minutely API request limit" in msg:
                    logger.warning("Rate limit hit, sleeping %ss before retrying %s",
                                   rate_limit_backoff_sec, name)
                    time.sleep(rate_limit_backoff_sec)
                    continue
                logger.error("Unrecoverable error fetching %s: %s", name, exc)
                break

    hourly_df = pd.concat(all_hourly, ignore_index=True) if all_hourly else pd.DataFrame()
    daily_df = pd.concat(all_daily, ignore_index=True) if all_daily else pd.DataFrame()
    return hourly_df, daily_df


def load_weather_from_disk(
    hourly_path: Optional[Path] = None,
    daily_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the already-downloaded hourly + daily Open-Meteo CSVs.

    This is the offline-friendly counterpart to :func:`fetch_weather_from_api`.
    """
    hourly_path = hourly_path or (RAW_OPENMETEO / "hourly_weather.csv")
    daily_path = daily_path or (RAW_OPENMETEO / "daily_weather.csv")

    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly weather CSV not found: {hourly_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily weather CSV not found: {daily_path}")

    logger.info("Reading hourly weather  <- %s", hourly_path)
    hourly = pd.read_csv(hourly_path, parse_dates=["date"])
    logger.info("Reading daily weather   <- %s", daily_path)
    daily = pd.read_csv(daily_path, parse_dates=["date"])

    # Normalise timezone — Open-Meteo stores tz-aware UTC timestamps
    for df in (hourly, daily):
        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")

    logger.info("Hourly: %s rows, %s cities, %s -> %s",
                f"{len(hourly):,}", hourly["City"].nunique(),
                hourly["date"].min(), hourly["date"].max())
    logger.info("Daily:  %s rows, %s cities, %s -> %s",
                f"{len(daily):,}", daily["City"].nunique(),
                daily["date"].min(), daily["date"].max())
    return hourly, daily


def validate_weather_df(df: pd.DataFrame, kind: str = "hourly") -> Dict[str, object]:
    """Return a dict of sanity-check metrics for a weather frame."""
    if df.empty:
        raise ValueError(f"{kind} weather frame is empty")

    required = {"City", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{kind} weather frame missing columns: {missing}")

    expected_step = pd.Timedelta(hours=1) if kind == "hourly" else pd.Timedelta(days=1)
    per_city_gaps: Dict[str, int] = {}
    for city, g in df.groupby("City"):
        diffs = g["date"].sort_values().diff().dropna()
        gaps = int((diffs != expected_step).sum())
        per_city_gaps[city] = gaps

    data_cols = df.drop(columns=["City", "date"])
    missing_cells = int(data_cols.isna().sum().sum())
    total_cells = int(data_cols.size)
    return {
        "rows": len(df),
        "cities": df["City"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "missing_cells": missing_cells,
        "missing_pct": (missing_cells / total_cells) if total_cells else 0.0,
        "timestep_gaps_per_city": per_city_gaps,
    }


# ============================================================================
# 2. POPULATION (per-city CSVs already rasterised from WorldPop TIFs)
# ============================================================================

def load_population(year_range: range = range(2020, 2027)) -> pd.DataFrame:
    """Stack the per-year population-density CSVs into a single long frame.

    Expected schema: ``City, Latitude, Longitude, Year, Pop_Density``.
    """
    frames: List[pd.DataFrame] = []
    for y in year_range:
        p = RAW_POPULATION / f"population_{y}.csv"
        if not p.exists():
            logger.warning("Population file missing for %s -> %s", y, p)
            continue
        frames.append(pd.read_csv(p))

    if not frames:
        raise FileNotFoundError(f"No population CSVs found in {RAW_POPULATION}")

    df = pd.concat(frames, ignore_index=True)
    expected = {"City", "Latitude", "Longitude", "Year", "Pop_Density"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Population data missing columns: {expected - set(df.columns)}")
    logger.info("Population: %d rows across years %s-%s, %d cities",
                len(df), df["Year"].min(), df["Year"].max(), df["City"].nunique())
    return df


# ============================================================================
# 3. ROADS / HUMAN ACCESS
# ============================================================================

def load_roads_static() -> pd.DataFrame:
    """Load the pre-computed 5 km road-density feature per city."""
    p = RAW_OSM_ROADS / "static_city_infrastructure.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Static road-density CSV not found: {p}. "
            "Generate it with extract_roads_from_pbf() first."
        )
    df = pd.read_csv(p)
    if not {"City", "human_access_road_meters"}.issubset(df.columns):
        raise ValueError("roads CSV must contain City and human_access_road_meters")
    logger.info("Roads: %d cities, total %.0f km",
                len(df), df["human_access_road_meters"].sum() / 1000)
    return df


def extract_roads_from_pbf(
    pbf_path: Path,
    cities: Optional[Dict[str, CityPoint]] = None,
    buffer_m: int = 5_000,
) -> pd.DataFrame:
    """Rebuild the per-city road-density feature from a national OSM PBF extract.

    This is the canonical, offline path (no web calls), using pyosmium for
    streaming extraction and geopandas for spatial clipping. It is slow to run
    (~90 s for Azerbaijan) so the notebook prefers the cached CSV via
    :func:`load_roads_static`.
    """
    try:
        import geopandas as gpd  # type: ignore
        import osmium  # type: ignore
        from shapely.geometry import LineString, Point  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "extract_roads_from_pbf() requires geopandas, osmium and shapely."
        ) from exc

    if not pbf_path.exists():
        raise FileNotFoundError(f"OSM PBF file not found: {pbf_path}")

    cities = cities or AZERBAIJAN_CITIES

    class _Handler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.roads: List[Dict] = []

        def way(self, w):  # noqa: D401
            if "highway" not in w.tags:
                return
            try:
                coords = [(n.lon, n.lat) for n in w.nodes]
                if len(coords) >= 2:
                    self.roads.append({"geometry": LineString(coords)})
            except osmium.InvalidLocationError:
                pass

    logger.info("Streaming %s ...", pbf_path)
    handler = _Handler()
    handler.apply_file(str(pbf_path), locations=True)
    logger.info("Parsed %d road segments", len(handler.roads))

    roads = gpd.GeoDataFrame(handler.roads, crs="EPSG:4326").to_crs(epsg=32639)

    rows = []
    for name, pt in cities.items():
        city_m = (
            gpd.GeoSeries([Point(pt.lon, pt.lat)], crs="EPSG:4326")
            .to_crs(epsg=32639)
            .iloc[0]
        )
        buf = city_m.buffer(buffer_m)
        local = roads.clip(buf)
        meters = float(local.geometry.length.sum())
        rows.append({"City": name, "human_access_road_meters": meters})
        logger.info("  %-12s %.0f m", name, meters)

    return pd.DataFrame(rows)


# ============================================================================
# 4. NDVI (Earth Engine export)
# ============================================================================

def load_ndvi() -> pd.DataFrame:
    """Load the per-city daily NDVI time series from Earth Engine."""
    p = RAW_EARTH_ENGINE / "aze_ndvi_2020_2026.csv"
    if not p.exists():
        raise FileNotFoundError(f"NDVI CSV not found: {p}")
    df = pd.read_csv(p)
    df = df.rename(columns={"city": "City"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df = df[["City", "date", "NDVI"]].sort_values(["City", "date"]).reset_index(drop=True)
    logger.info("NDVI: %d rows, %d cities, %s -> %s",
                len(df), df["City"].nunique(), df["date"].min(), df["date"].max())
    return df


# ============================================================================
# 5. NASA FIRMS fire hotspots
# ============================================================================

def load_firms() -> pd.DataFrame:
    """Stack every annual VIIRS FIRMS CSV into a single fire-hotspot frame."""
    files = sorted(RAW_NASA_FIRMS.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No FIRMS CSVs in {RAW_NASA_FIRMS}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    # Combine acq_date + 4-digit acq_time (e.g. 850 -> 08:50 UTC)
    hhmm = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["acq_timestamp"] = pd.to_datetime(
        df["acq_date"].dt.strftime("%Y-%m-%d") + " " + hhmm.str[:2] + ":" + hhmm.str[2:],
        utc=True,
    )
    logger.info("FIRMS: %d hotspots, %s -> %s",
                len(df), df["acq_date"].min(), df["acq_date"].max())
    return df


# ============================================================================
# 6. LIGHTNING
# ============================================================================

def load_lightning() -> pd.DataFrame:
    """Stack every annual lightning CSV (lat, lon, month, thunder_hours)."""
    files = sorted(RAW_LIGHTNING.glob("azerbaijan_lightning_*.csv"))
    if not files:
        raise FileNotFoundError(f"No lightning CSVs in {RAW_LIGHTNING}")

    rows: List[pd.DataFrame] = []
    for f in files:
        year = int(f.stem.split("_")[-1])
        d = pd.read_csv(f)
        d["year"] = year
        rows.append(d)

    df = pd.concat(rows, ignore_index=True)
    expected = {"lat", "lon", "month", "thunder_hours", "year"}
    if not expected.issubset(df.columns):
        raise ValueError(f"lightning frame missing: {expected - set(df.columns)}")
    logger.info("Lightning: %d grid-month obs over %d years",
                len(df), df["year"].nunique())
    return df


# ============================================================================
# 7. FOREST BOUNDARIES (KMZ -> KML -> GeoDataFrame)
# ============================================================================

def load_forest_boundaries():
    """Return forest polygons as a :class:`geopandas.GeoDataFrame`.

    Unpacks the KMZ into ``data/raw/forest_boundaries/extracted/`` the first
    time it is called, then reads the KML. Requires geopandas + fiona.
    """
    try:
        import geopandas as gpd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("load_forest_boundaries() requires geopandas.") from exc

    extract_dir = RAW_FOREST_BOUNDARIES / "extracted"
    kmz_path = RAW_FOREST_BOUNDARIES / "azerbaijan.kmz"
    kml_path: Optional[Path] = None

    if extract_dir.exists():
        kmls = list(extract_dir.glob("*.kml"))
        kml_path = kmls[0] if kmls else None

    if kml_path is None:
        if not kmz_path.exists():
            raise FileNotFoundError(f"Forest KMZ not found: {kmz_path}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(kmz_path, "r") as z:
            z.extractall(extract_dir)
        kmls = list(extract_dir.glob("*.kml"))
        if not kmls:
            raise RuntimeError(f"No KML inside {kmz_path}")
        kml_path = kmls[0]

    # Enable KML driver explicitly (fiona ships it but keeps it off by default)
    try:
        import fiona  # type: ignore

        fiona.drvsupport.supported_drivers["KML"] = "rw"
        fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
    except Exception:  # noqa: BLE001
        pass

    gdf = gpd.read_file(kml_path)
    logger.info("Forests: %d features from %s", len(gdf), kml_path.name)
    return gdf


# ============================================================================
# 8. LANDCOVER (sampled from a Copernicus / ESA TIF at each city)
# ============================================================================

def sample_landcover_at_cities(
    tif_path: Optional[Path] = None,
    cities: Optional[Dict[str, CityPoint]] = None,
) -> pd.DataFrame:
    """Return one landcover class per city by sampling a raster at its centroid."""
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise RuntimeError("sample_landcover_at_cities() requires rasterio.") from exc

    tif_path = tif_path or (RAW_EARTH_ENGINE / "aze_landcover.tif")
    cities = cities or AZERBAIJAN_CITIES
    if not tif_path.exists():
        raise FileNotFoundError(f"Landcover TIF not found: {tif_path}")

    rows: List[Dict] = []
    with rasterio.open(tif_path) as src:
        band = src.read(1)
        h, w = src.height, src.width
        for name, pt in cities.items():
            r, c = src.index(pt.lon, pt.lat)
            val = int(band[r, c]) if 0 <= r < h and 0 <= c < w else 0
            rows.append({"City": name, "Landcover_Class": val})

    df = pd.DataFrame(rows)
    logger.info("Landcover: %d cities sampled, classes = %s",
                len(df), sorted(df["Landcover_Class"].unique().tolist()))
    return df


# ============================================================================
# 9. PERSISTENCE
# ============================================================================

def save_interim(df: pd.DataFrame, name: str, fmt: str = "csv") -> Path:
    """Save a DataFrame to ``data/interim/<name>.<fmt>``.

    ``fmt`` may be ``'csv'`` or ``'parquet'``. Parquet is preferred in production
    but requires pyarrow/fastparquet; CSV is the portable fallback.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out = INTERIM_DIR / f"{name}.{fmt}"
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "parquet":
        df.to_parquet(out, index=False)
    else:
        raise ValueError(f"Unsupported fmt {fmt!r}")
    logger.info("Saved %s rows -> %s (%.2f MB)",
                f"{len(df):,}", out, out.stat().st_size / 1024 / 1024)
    return out


# ============================================================================
# 10. ORCHESTRATOR
# ============================================================================

def run_ingestion_pipeline(
    use_api: bool = False,
    fmt: str = "csv",
    include_geo: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run every loader and persist the results to ``data/interim/``.

    Parameters
    ----------
    use_api
        If ``True``, attempt a fresh Open-Meteo download; otherwise read the
        cached CSVs on disk.
    fmt
        ``'csv'`` or ``'parquet'`` for the interim artefacts.
    include_geo
        Skip geopandas / rasterio steps (forests, landcover) when they would
        fail because optional geo libraries are missing.

    Returns
    -------
    dict
        Name -> DataFrame / GeoDataFrame for every source successfully loaded.
    """
    logger.info("=" * 72)
    logger.info("PHASE 1 - Data Ingestion pipeline starting")
    logger.info("=" * 72)

    outputs: Dict[str, pd.DataFrame] = {}

    # --- Weather -----------------------------------------------------------
    if use_api:
        try:
            hourly, daily = fetch_weather_from_api()
        except RuntimeError as exc:
            logger.warning("API fetch unavailable (%s) - falling back to disk", exc)
            hourly, daily = load_weather_from_disk()
    else:
        hourly, daily = load_weather_from_disk()
    outputs["weather_hourly"] = hourly
    outputs["weather_daily"] = daily
    save_interim(hourly, "weather_hourly_raw", fmt=fmt)
    save_interim(daily, "weather_daily_raw", fmt=fmt)

    # --- Cities reference table -------------------------------------------
    cities_df = pd.DataFrame(cities_as_records())
    outputs["cities"] = cities_df
    save_interim(cities_df, "cities_reference", fmt=fmt)

    # --- Auxiliary sources (best-effort; log and continue on failure) -----
    for key, loader in [
        ("population", load_population),
        ("roads", load_roads_static),
        ("ndvi", load_ndvi),
        ("firms", load_firms),
        ("lightning", load_lightning),
    ]:
        try:
            df = loader()
            outputs[key] = df
            save_interim(df, key, fmt=fmt)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping %s: %s", key, exc)

    # --- Optional geo sources ---------------------------------------------
    if include_geo:
        try:
            forests = load_forest_boundaries()
            outputs["forests"] = forests
            forests_path = INTERIM_DIR / "forest_boundaries.geojson"
            forests.to_file(forests_path, driver="GeoJSON")
            logger.info("Saved %d forest polygons -> %s", len(forests), forests_path)
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("Skipping forests: %s", exc)

        try:
            lc = sample_landcover_at_cities()
            outputs["landcover"] = lc
            save_interim(lc, "landcover_at_cities", fmt=fmt)
        except (RuntimeError, FileNotFoundError) as exc:
            logger.error("Skipping landcover: %s", exc)

    logger.info("=" * 72)
    logger.info("Ingestion complete - %d datasets prepared in %s",
                len(outputs), INTERIM_DIR)
    logger.info("=" * 72)
    return outputs


if __name__ == "__main__":
    run_ingestion_pipeline()
