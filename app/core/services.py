"""Core services for launch window calculations (public API sources)."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import aiohttp
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from prometheus_client import Counter, Histogram

from app.core.config import settings
from app.schemas.launch import LaunchWindow, WeatherCondition, SpaceWeather
from app.ml.anomaly_detector import AnomalyDetector

# Metrics
FETCH_FAILURES = Counter("fetch_failures_total", "Number of API fetch failures", ["source"])
RESPONSE_TIME = Histogram("response_time_seconds", "Response time in seconds", ["endpoint"])


class LaunchService:
    """Service for managing launch windows, weather, space weather and anomalies."""

    def __init__(self) -> None:
        self.anomaly_detector = AnomalyDetector()
        self.session: Optional[aiohttp.ClientSession] = None

    async def get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    # --------------------- Fetchers ---------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def fetch_tle_data(self) -> List[Dict[str, Any]]:
        session = await self.get_session()
        try:
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def fetch_weather_data(self, site: str) -> WeatherCondition:
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
            return WeatherCondition(
                wind_speed=wind_speed_knots,
                wind_direction=0.0,
                temperature=temperature_f,
                pressure=pressure,
            )
        except Exception:
            FETCH_FAILURES.labels(source="nws_weather").inc()
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def fetch_space_weather(self) -> SpaceWeather:
        session = await self.get_session()
        try:
            async with session.get(settings.NOAA_SPACE_WEATHER_URL) as resp:
                data = await resp.json()
            latest = data[-1]
            return SpaceWeather(
                kp_index=float(latest["kp"]),
                solar_wind_speed=float(latest.get("speed", 0)),
                magnetic_field_bt=float(latest.get("bt", 0)),
                timestamp=datetime.utcnow(),
            )
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
        import torch as _torch

        tensor = _torch.tensor(
            [[len(t["line1"]), len(t["line2"])] for t in tle_sets], dtype=_torch.float32
        )
        scores = self.anomaly_detector.detect(tensor)
        out: List[Dict[str, Any]] = []
        for idx, score in enumerate(scores):
            if score > self.anomaly_detector.threshold:
                out.append(
                    {
                        "type": "orbital_anomaly",
                        "description": f"Anomalous TLE structure detected for {tle_sets[idx]['name']}",
                        "severity": int(score * 10),
                        "detected_at": datetime.utcnow(),
                        "is_active": True,
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
