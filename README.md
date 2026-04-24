# Dashboard & API

FastAPI backend + vanilla-JS dashboard for serving the pipeline's outputs.

## Run it

From the project root:

```bash
# 1. Install the web deps
pip install fastapi "uvicorn[standard]" pydantic

# 2. Make sure the pipeline has produced the artefacts
#    (run notebooks 01-09 once, or:)
python -m src.weather.forecast
python -m src.climate.trends
python -m src.wildfire.predict

# 3. Start the API + dashboard
uvicorn app.api.main:app --reload --port 8000
```

Then open:

- **Dashboard**: http://localhost:8000/
- **Swagger / OpenAPI docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Architecture

```
app/
├── api/
│   ├── main.py       # FastAPI app + routes + Pydantic models
│   └── service.py    # DataService: all pandas lives here
└── static/
    ├── index.html    # Dashboard (served at /)
    └── preview.html  # Offline preview (reads preview_data/*.json)
    └── preview_data/ # JSON snapshot of every endpoint
```

**Layering**: `main.py` declares routes. `service.py` does the data work.
This split means every query is testable without spinning up uvicorn, and
swapping pandas for DuckDB later changes one class.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/meta` | Anchor date, cities, targets, generated-at |
| GET | `/weather` | Wide-format 30-day forecast (one row per city-day) |
| GET | `/weather/long` | Long-format forecast (one row per city-day-target) |
| GET | `/wildfire` | 30-day fire risk per city-day |
| GET | `/wildfire/summary` | Per-city rollup of 30-day risk |
| GET | `/insights` | Combined per-city headlines |
| GET | `/insights/trends` | Annual Mann-Kendall + Theil-Sen per city × variable |
| GET | `/insights/anomalies` | Forecast vs climatology z-scores |
| GET | `/insights/climatology` | Daily DOY climatology for one city + variable |
| POST | `/admin/refresh` | Reload artefacts from disk (no restart needed) |

Every endpoint supports standard query parameters (`city=`, `target=`, etc.)
and returns JSON that maps directly to the Pydantic response models you'll
see in `/docs`.

## The preview mode

`preview.html` in `app/static/` loads its data from `preview_data/*.json`
instead of the API. It's a self-contained offline demo of what the live
dashboard looks like — just open it in a browser after running the pipeline
once. No server, no install. Handy for screenshots in pull requests.

## Production deploy

For a real deploy, swap these defaults:

1. **CORS**: set `allow_origins=["https://your-domain.tld"]` (currently `["*"]`).
2. **Caching**: add Redis or `fastapi-cache2` in front of the large endpoints.
3. **Refresh**: replace `POST /admin/refresh` with a scheduled cron/Celery
   job that re-runs the pipeline and then hits the endpoint internally.
4. **Static frontend**: move `app/static/*` to S3/CloudFront or similar CDN.
5. **Observability**: add a Prometheus middleware + structured logs (use
   `structlog` or `logfire`).
6. **Auth**: none today; add OAuth2 or an API-key dependency on `/admin/*`.
