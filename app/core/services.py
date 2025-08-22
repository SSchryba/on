"""Core services for launch window calculations (public API sources)."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib

import aiohttp
import torch
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from prometheus_client import Counter, Histogram
from cachetools import TTLCache, cached
from cachetools.keys import hashkey

from app.core.config import settings
from app.schemas.launch import LaunchWindow, WeatherCondition, SpaceWeather
from app.ml.anomaly_detector import OptimizedAnomalyDetector

# Metrics
FETCH_FAILURES = Counter("fetch_failures_total", "Number of API fetch failures", ["source"])
RESPONSE_TIME = Histogram("response_time_seconds", "Response time in seconds", ["endpoint"])
CACHE_HITS = Counter("cache_hits_total", "Number of cache hits", ["source"])
CACHE_MISSES = Counter("cache_misses_total", "Number of cache misses", ["source"])

# Caches with different TTLs based on data freshness
tle_cache = TTLCache(maxsize=50, ttl=86400)  # 24 hours for TLE data
weather_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour for weather forecasts
space_weather_cache = TTLCache(maxsize=100, ttl=1800)  # 30 minutes for space weather

def cache_key_for_weather(site: str) -> str:
    """Generate cache key for weather data."""
    return f"weather_{site}"

def cache_key_for_space_weather() -> str:
    """Generate cache key for space weather data."""
    return "space_weather"


class LaunchService:
    """Service for managing launch windows, weather, space weather and anomalies."""

    def __init__(self) -> None:
        self.anomaly_detector = OptimizedAnomalyDetector()
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    # --------------------- Fetchers ---------------------
    @cached(tle_cache, key=lambda self: "tle_data")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), 
           retry=retry_if_exception_type(aiohttp.ClientError))
    async def fetch_tle_data(self) -> List[Dict[str, Any]]:
        """Fetch and parse public TLE data from Celestrak (cached for 24h)."""
        try:
            CACHE_MISSES.labels(source="celestrak").inc()
            session = await self.get_session()
            async with session.get(settings.CELESTRAK_TLE_URL) as resp:
                text = await resp.text()
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            sets: List[Dict[str, str]] = []
            for i in range(0, len(lines) - 2, 3):
                name, l1, l2 = lines[i : i + 3]
                sets.append({"name": name, "line1": l1, "line2": l2})
            return sets[:50]
        except Exception:
            FETCH_FAILURES.labels(source="celestrak").inc()
            raise
        else:
            CACHE_HITS.labels(source="celestrak").inc()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(aiohttp.ClientError))
    async def fetch_weather_data(self, site: str) -> WeatherCondition:
        """Fetch weather via NOAA NWS API with caching."""
        cache_key = cache_key_for_weather(site)
        
        # Check cache first
        if cache_key in weather_cache:
            CACHE_HITS.labels(source="nws_weather").inc()
            return weather_cache[cache_key]
        
        CACHE_MISSES.labels(source="nws_weather").inc()
        session = await self.get_session()
        site_data = settings.LAUNCH_SITES.get(site)
        if not site_data:
            raise ValueError(f"Unknown launch site: {site}")
        lat, lon = site_data["lat"], site_data["lon"]
        points_url = f"{settings.NWS_POINTS_URL}/{lat},{lon}"
        try:
            async with session.get(points_url, headers={"User-Agent": "SpaceLaunchTracker/1.0"}) as resp:
                meta = await resp.json()
            forecast_url = meta["properties"]["forecast"]
            async with session.get(forecast_url, headers={"User-Agent": "SpaceLaunchTracker/1.0"}) as resp:
                forecast = await resp.json()
            period = forecast["properties"]["periods"][0]
            wind_speed_mph = float(period["windSpeed"].split()[0]) if period.get("windSpeed") else 0.0
            wind_speed_knots = wind_speed_mph * 0.868976
            temperature_f = float(period.get("temperature", 0.0))
            pressure = 1013.25
            
            weather_condition = WeatherCondition(
                wind_speed=wind_speed_knots,
                wind_direction=0.0,
                temperature=temperature_f,
                pressure=pressure,
            )
            
            # Cache the result
            weather_cache[cache_key] = weather_condition
            return weather_condition
        except Exception:
            FETCH_FAILURES.labels(source="nws_weather").inc()
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8),
           retry=retry_if_exception_type(aiohttp.ClientError))
    async def fetch_space_weather(self) -> SpaceWeather:
        """Fetch space weather data with caching."""
        cache_key = cache_key_for_space_weather()
        
        # Check cache first
        if cache_key in space_weather_cache:
            CACHE_HITS.labels(source="space_weather").inc()
            return space_weather_cache[cache_key]
        
        CACHE_MISSES.labels(source="space_weather").inc()
        session = await self.get_session()
        try:
            async with session.get(settings.NOAA_SPACE_WEATHER_URL) as resp:
                data = await resp.json()
            latest = data[-1]
            
            space_weather = SpaceWeather(
                kp_index=float(latest["kp"]),
                solar_wind_speed=float(latest.get("speed", 0)),
                magnetic_field_bt=float(latest.get("bt", 0)),
                timestamp=datetime.utcnow(),
            )
            
            # Cache the result
            space_weather_cache[cache_key] = space_weather
            return space_weather
        except Exception:
            FETCH_FAILURES.labels(source="space_weather").inc()
            raise

    # --------------------- Business logic ---------------------
    async def get_launch_windows(self, site: str) -> List[LaunchWindow]:
        weather = await self.fetch_weather_data(site)
        space_weather = await self.fetch_space_weather()
        _ = await self.fetch_tle_data()  # currently unused placeholder for future orbital constraints

        constraints: List[str] = []
        if weather.wind_speed > settings.MAX_WIND_SPEED_KNOTS:
            constraints.append("Wind speed exceeds maximum")
        if space_weather.kp_index > settings.MAX_KP_INDEX:
            constraints.append("Geomagnetic activity too high")

        windows: List[LaunchWindow] = []
        now = datetime.utcnow()
        for h in range(24):
            start = now + timedelta(hours=h)
            end = start + timedelta(hours=1)
            windows.append(
                LaunchWindow(
                    start_time=start,
                    end_time=end,
                    site=site,
                    weather=weather,
                    space_weather=space_weather,
                    is_viable=len(constraints) == 0,
                    constraints=constraints,
                )
            )
        return windows

    async def get_anomalies(self) -> List[Dict[str, Any]]:
        tle_sets = await self.fetch_tle_data()
        if not tle_sets:
            return []

        # Use optimized detector with actual TLE data
        scores = self.anomaly_detector.detect(tle_sets)
        threshold = self.anomaly_detector.threshold or 0.8  # fallback threshold
        
        out: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if score > threshold:
                out.append(
                    {
                        "type": "orbital_anomaly",
                        "description": f"Anomalous TLE characteristics detected for {tle_sets[idx]['name']}",
                        "severity": min(10, int(score * 10)),
                        "detected_at": datetime.utcnow(),
                        "is_active": True,
                        "anomaly_score": round(score, 3),
                    }
                )
        return out

    # --------------------- Helper wrappers ---------------------
    async def get_launch_sites(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "latitude": cfg["lat"], "longitude": cfg["lon"]}
            for name, cfg in settings.LAUNCH_SITES.items()
        ]

    async def get_space_weather(self) -> Dict[str, Any]:  # wrapper for API route
        sw = await self.fetch_space_weather()
        return sw.model_dump() if hasattr(sw, "model_dump") else sw.__dict__


launch_service = LaunchService()
