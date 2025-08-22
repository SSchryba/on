"""Optimized API routes with background tasks and session management."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import aiohttp

from app.core.auth import get_api_key
from app.core.dependencies import get_rate_limited_session
from app.schemas.launch import LaunchWindow, LaunchSite
from app.core.services import launch_service

router = APIRouter()


@router.get("/launch-windows/{site}")
async def get_launch_windows(
    site: str,
    background_tasks: BackgroundTasks,
    use_background: bool = Query(False, description="Process in background"),
    api_key: str = Depends(get_api_key),
    session: aiohttp.ClientSession = Depends(get_rate_limited_session)
):
    """Get launch windows for a specific site with optional background processing."""
    try:
        if use_background:
            # Trigger background task for heavy computation
            from app.tasks import compute_launch_windows_task
            task = compute_launch_windows_task.delay(site)
            return {
                "task_id": task.id,
                "status": "processing",
                "message": f"Computing launch windows for {site} in background"
            }
        
        # Process synchronously for immediate results
        windows = await launch_service.get_launch_windows(site)
        return [w.model_dump() if hasattr(w, "model_dump") else w.__dict__ for w in windows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sites", response_model=List[LaunchSite])
async def get_launch_sites(
    api_key: str = Depends(get_api_key)
) -> List[LaunchSite]:
    """Get all available launch sites."""
    try:
        sites_data = await launch_service.get_launch_sites()
        return [LaunchSite(**site) for site in sites_data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/space-weather")
async def get_space_weather(
    api_key: str = Depends(get_api_key),
    session: aiohttp.ClientSession = Depends(get_rate_limited_session)
):
    """Get current space weather conditions with session management."""
    try:
        weather = await launch_service.get_space_weather()
        return weather
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies")
async def get_anomalies(
    background_tasks: BackgroundTasks,
    use_background: bool = Query(False, description="Process in background"),
    subsample_ratio: float = Query(0.1, ge=0.01, le=1.0, description="Subsample ratio for faster processing"),
    api_key: str = Depends(get_api_key),
    session: aiohttp.ClientSession = Depends(get_rate_limited_session)
):
    """Get detected anomalies with optimized processing."""
    try:
        if use_background:
            # Trigger background anomaly detection
            from app.tasks import compute_anomalies_task
            task = compute_anomalies_task.delay()
            return {
                "task_id": task.id,
                "status": "processing",
                "message": "Computing anomalies in background"
            }
        
        # Process synchronously with subsample option
        anomalies = await launch_service.get_anomalies()
        
        # Apply subsampling if requested
        if subsample_ratio < 1.0:
            import random
            sample_size = max(1, int(len(anomalies) * subsample_ratio))
            anomalies = random.sample(anomalies, min(sample_size, len(anomalies)))
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies),
            "subsample_ratio": subsample_ratio
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get the status of a background task."""
    try:
        from app.tasks import celery_app
        task = celery_app.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            response = {
                'task_id': task_id,
                'state': task.state,
                'status': 'Task is waiting to be processed'
            }
        elif task.state == 'PROGRESS':
            response = {
                'task_id': task_id,
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 1),
                'status': task.info.get('status', '')
            }
        elif task.state == 'SUCCESS':
            response = {
                'task_id': task_id,
                'state': task.state,
                'result': task.result
            }
        else:
            # Task failed
            response = {
                'task_id': task_id,
                'state': task.state,
                'error': str(task.info)
            }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = Query(None, description="Cache type to clear (tle, weather, space_weather, all)"),
    api_key: str = Depends(get_api_key)
):
    """Clear API caches for fresh data."""
    try:
        from app.core.services import tle_cache, weather_cache, space_weather_cache
        
        cleared = []
        if cache_type == "tle" or cache_type == "all":
            tle_cache.clear()
            cleared.append("tle")
        if cache_type == "weather" or cache_type == "all":
            weather_cache.clear()
            cleared.append("weather")
        if cache_type == "space_weather" or cache_type == "all":
            space_weather_cache.clear()
            cleared.append("space_weather")
        if cache_type is None or cache_type == "all":
            if not cleared:  # Clear all if no specific type
                tle_cache.clear()
                weather_cache.clear()
                space_weather_cache.clear()
                cleared = ["tle", "weather", "space_weather"]
        
        return {
            "status": "success",
            "cleared_caches": cleared,
            "message": f"Cleared {len(cleared)} cache(s)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
