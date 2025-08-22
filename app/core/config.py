"""Application configuration (production hardened)."""

from typing import Any, Dict, List
from pydantic_settings import BaseSettings
from pydantic import field_validator, model_validator
import json
import os


class Settings(BaseSettings):
    # Metadata
    PROJECT_NAME: str = "SpaceLaunchTracker"
    PROJECT_DESCRIPTION: str = "Space traffic and launch window tracking service (public data sources)"
    VERSION: str = "1.2.0"
    PRODUCTION: bool = True  # default to production stance; override locally if needed

    # Public data sources
    CELESTRAK_TLE_URL: str = "https://celestrak.org/NORAD/elements/active.txt"
    NOAA_SPACE_WEATHER_URL: str = "https://services.swpc.noaa.gov/products/geospace/planetary-k-index-forecast.json"
    NWS_POINTS_URL: str = "https://api.weather.gov/points"

    # Security / Auth
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = ""  # MUST be provided via environment / secret store
    ALLOWED_ORIGINS: List[str] = ["*"]  # tighten in deployment

    # Caching / Redis
    REDIS_URL: str = "redis://redis:6379/0"
    CACHE_TTL: int = 300

    # Thresholds
    MAX_WIND_SPEED_KNOTS: float = 20.0
    MAX_KP_INDEX: int = 5

    # Launch sites (real coordinates)
    LAUNCH_SITES: Dict[str, Dict[str, float]] = {
        "cape_canaveral": {"lat": 28.396837, "lon": -80.605659},
        "vandenberg": {"lat": 34.742, "lon": -120.5724},
    }
    LAUNCH_SITES_JSON: str | None = None

    # Feature flags
    ENABLE_ANOMALY_DETECTION: bool = True
    ENABLE_WEBSOCKETS: bool = True
    ENABLE_METRICS: bool = True

    SCHEDULE_FETCH_CRON: str = "*/15 * * * *"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @field_validator("LAUNCH_SITES", mode="before")
    @classmethod
    def merge_launch_sites(cls, v: Any) -> Dict[str, Any]:
        env_json = os.getenv("LAUNCH_SITES_JSON")
        if env_json:
            try:
                parsed = json.loads(env_json)
                if isinstance(parsed, dict):
                    base = dict(v) if isinstance(v, dict) else {}
                    base.update(parsed)
                    return base
            except json.JSONDecodeError:
                return v
        return v

    @model_validator(mode="after")
    def enforce_api_key(self) -> "Settings":  # type: ignore[name-defined]
        if not self.API_KEY:
            raise ValueError("API_KEY must be set (no default dev key in production-hardened config)")
        return self


settings = Settings()