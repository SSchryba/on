"""API routes for the application."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.core.auth import get_api_key
from app.schemas.launch import LaunchWindow, LaunchSite
from app.core.services import launch_service

router = APIRouter()

@router.get("/launch-windows/{site}", response_model=List[LaunchWindow])
async def get_launch_windows(
    site: str,
    api_key: str = Depends(get_api_key)
) -> List[LaunchWindow]:
    """Get launch windows for a specific site."""
    try:
        windows = await launch_service.get_launch_windows(site)
        return windows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sites", response_model=List[LaunchSite])
async def get_launch_sites(
    api_key: str = Depends(get_api_key)
) -> List[LaunchSite]:
    """Get all available launch sites."""
    try:
        sites = await launch_service.get_launch_sites()
        return sites
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/space-weather")
async def get_space_weather(
    api_key: str = Depends(get_api_key)
):
    """Get current space weather conditions."""
    try:
        weather = await launch_service.get_space_weather()
        return weather
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies")
async def get_anomalies(
    api_key: str = Depends(get_api_key)
):
    """Get detected anomalies in space traffic."""
    try:
        anomalies = await launch_service.get_anomalies()
        return anomalies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
