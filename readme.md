## SpaceLaunchTracker

### Overview
SpaceLaunchTracker aggregates public space situational awareness (SSA) and environmental data to project orbital states, assess constraints, and derive candidate launch windows. This edition uses ONLY fully public, keyless APIs for zero-friction deployment.

### Public Data Sources (No Credentials Required)
| Domain | Source | Endpoint Example | Notes |
|--------|--------|------------------|-------|
| Orbital TLE | Celestrak | https://celestrak.org/NORAD/elements/active.txt | Bulk active satellite TLEs (parsed in triplets) |
| Weather (forecast) | NOAA NWS API | https://api.weather.gov/points/{lat},{lon} -> forecast | Requires User-Agent header, no key |
| Space Weather (Kp) | NOAA SWPC | https://services.swpc.noaa.gov/products/geospace/planetary-k-index-forecast.json | Latest forecast row used |

### Key Features
- Async fetching (aiohttp) with retries (tenacity)
- Launch window generation with weather + space weather constraints
- Autoencoder anomaly detection on TLE structural variation
- FastAPI REST API + Prometheus metrics + internal API key auth
- Docker & docker-compose for containerized deployment
- Ready for Celery/Redis background tasks

### Install & Run (No API Keys Needed)
```
git clone https://github.com/yourusername/SpaceLaunchTracker.git
cd SpaceLaunchTracker
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
copy .env.robust.template .env  # Fill API_KEY (internal only)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Docker:
```
docker compose up --build -d
```

### Minimal .env
```
API_KEY=replace_me
PRODUCTION=false
```

### API Endpoints
- GET /health – liveness
- GET /api/v1/launch-windows/{site} – 24h rolling windows (header: X-API-Key)
- GET /api/v1/sites – available launch sites
- GET /api/v1/space-weather – current Kp snapshot
- GET /api/v1/anomalies – anomaly detections from recent TLE sample

### Launch Window Constraints
1. Wind speed (knots) <= MAX_WIND_SPEED_KNOTS
2. Kp index <= MAX_KP_INDEX
If constraints pass, each hourly slot in next 24h is viable.

### TLE Parsing
Triplets (name, line1, line2) parsed; simple numeric features (line lengths) used for anomaly heuristics. Future: integrate precise orbital propagation with Skyfield / Poliastro.

### Anomaly Detection
Light autoencoder trains on first use; reconstruction error threshold flags outliers. Replace with domain-calibrated model for production-grade scoring.

### Testing
```
pytest --maxfail=1 -q
```
Mock aiohttp calls for deterministic tests (extend as needed).

### Configuration
Override / append launch sites via LAUNCH_SITES_JSON. Adjust thresholds in env.

### Metrics
Prometheus metrics at /metrics (if ENABLE_METRICS=true).

### Security
Internal API key only. Add OAuth2/JWT for broader exposure as needed.

### Roadmap
- Orbital propagation & conjunction analysis
- NWS gridpoint pressure & wind direction decoding
- Persistent DB + historical analytics
- WebSocket push for live updates

### License
MIT

### Changelog
- v1.1.0: Migrated to fully public APIs (Celestrak, NWS, NOAA SWPC); removed credential dependencies.
- v1.0.0: Initial production-ready release (credentialed APIs).