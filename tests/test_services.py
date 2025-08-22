"""Tests for the core services."""

import pytest
from datetime import datetime
import torch

from app.core.services import LaunchService
from app.schemas.launch import WeatherCondition, SpaceWeather, LaunchWindow

@pytest.mark.asyncio
async def test_fetch_space_track_data(launch_service):
    """Test fetching Space-Track data."""
    data = await launch_service.fetch_space_track_data()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "NORAD_CAT_ID" in data[0]

@pytest.mark.asyncio
async def test_fetch_weather_data(launch_service):
    """Test fetching weather data."""
    weather = await launch_service.fetch_weather_data("cape_canaveral")
    assert isinstance(weather, WeatherCondition)
    assert hasattr(weather, "wind_speed")
    assert hasattr(weather, "temperature")

@pytest.mark.asyncio
async def test_fetch_space_weather(launch_service):
    """Test fetching space weather data."""
    weather = await launch_service.fetch_space_weather()
    assert isinstance(weather, SpaceWeather)
    assert hasattr(weather, "kp_index")
    assert hasattr(weather, "timestamp")

@pytest.mark.asyncio
async def test_get_launch_windows(launch_service):
    """Test getting launch windows."""
    windows = await launch_service.get_launch_windows("cape_canaveral")
    assert isinstance(windows, list)
    assert len(windows) > 0
    assert all(isinstance(w, LaunchWindow) for w in windows)

@pytest.mark.asyncio
async def test_get_anomalies(launch_service):
    """Test anomaly detection."""
    anomalies = await launch_service.get_anomalies()
    assert isinstance(anomalies, list)
    if anomalies:
        assert all(isinstance(a["type"], str) for a in anomalies)
        assert all(isinstance(a["severity"], int) for a in anomalies)
        assert all(isinstance(a["detected_at"], datetime) for a in anomalies)
