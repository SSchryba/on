# analysis.py - Projections, anomalies, launch windows

from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
from typing import List, Dict, Any
from app.core.config import settings


def project_orbit(tle_data: List[Dict[str, str]]):
    """Project orbit 1 hour ahead using a placeholder circular orbit.

    (Future enhancement: integrate with skyfield / poliastro TLE parsing.)
    """
    ss = Orbit.circular(Earth, alt=400 * u.km)
    future_pos = ss.propagate(1 * u.h)
    return future_pos


def detect_anomalies(space_weather: List[Dict[str, Any]]):
    """Detect anomalies in Kp index sequence via simple z-score >3 rule."""
    if not space_weather:
        return np.array([])
    # Extract kp values; NOAA forecast rows vary—use 'kp' key if present else 0
    kp_values = []
    for row in space_weather:
        val = row.get('kp') or row.get('Kp') or row.get('kp_index')
        try:
            kp_values.append(float(val))
        except (TypeError, ValueError):
            kp_values.append(0.0)
    arr = np.array(kp_values)
    if arr.size < 3:
        return np.array([])
    mean, std = arr.mean(), arr.std() or 1.0
    anomalies = arr[np.abs(arr - mean) > 3 * std]
    return anomalies


def define_launch_window(weather_data: Dict[str, Any], space_weather: List[Dict[str, Any]], orbit_projection):
    wind_speed = weather_data.get('wind', {}).get('speed', 0)
    latest_kp = 0
    if space_weather:
        last = space_weather[-1]
        for key in ('kp', 'Kp', 'kp_index'):
            if key in last:
                try:
                    latest_kp = float(last[key])
                except (TypeError, ValueError):
                    latest_kp = 0
                break

    if wind_speed <= settings.MAX_WIND_SPEED_KNOTS and latest_kp <= settings.MAX_KP_INDEX:
        return f"Launch window open (projected safe orbit: {orbit_projection})"
    return "Launch window closed due to weather/space conditions"