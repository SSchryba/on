"""Tests for the API endpoints."""

import pytest
from fastapi import status
import os

def test_health_check(test_client):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}

def test_launch_windows_endpoint(test_client):
    """Test getting launch windows."""
    api_key = os.getenv("API_KEY", "")
    response = test_client.get(
        "/api/v1/launch-windows/cape_canaveral",
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    window = data[0]
    assert "start_time" in window
    assert "end_time" in window
    assert "is_viable" in window

def test_space_weather_endpoint(test_client):
    """Test getting space weather data."""
    api_key = os.getenv("API_KEY", "")
    response = test_client.get(
        "/api/v1/space-weather",
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "kp_index" in data
    assert isinstance(data["kp_index"], float)

def test_anomalies_endpoint(test_client):
    """Test getting anomalies."""
    api_key = os.getenv("API_KEY", "")
    response = test_client.get(
        "/api/v1/anomalies",
        headers={"X-API-Key": api_key}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)

def test_invalid_api_key(test_client):
    """Test endpoint with invalid API key."""
    response = test_client.get(
        "/api/v1/launch-windows/cape_canaveral",
        headers={"X-API-Key": "invalid_key"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN
