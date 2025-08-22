"""Background tasks using Celery for non-blocking operations."""

from typing import Dict, Any, List
from celery import Celery
from app.core.config import settings
from app.core.services import LaunchService

# Configure Celery
celery_app = Celery(
    "spacelaunch_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    timezone="UTC",
    enable_utc=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    beat_schedule={
        "refresh-space-data": {
            "task": "app.tasks.refresh_all_space_data",
            "schedule": 900.0,  # Every 15 minutes
        },
    },
)


@celery_app.task(name="tasks.health.ping")
def ping() -> str:
    """Health check task."""
    return "pong"


@celery_app.task(name="tasks.compute_launch_windows")
def compute_launch_windows_task(site: str) -> Dict[str, Any]:
    """Compute launch windows for a site in background."""
    import asyncio
    
    async def _compute():
        service = LaunchService()
        try:
            windows = await service.get_launch_windows(site)
            return {
                "site": site,
                "windows": [w.model_dump() if hasattr(w, "model_dump") else w.__dict__ for w in windows],
                "status": "success"
            }
        except Exception as e:
            return {
                "site": site,
                "error": str(e),
                "status": "error"
            }
        finally:
            if service.session and not service.session.closed:
                await service.session.close()
    
    return asyncio.run(_compute())


@celery_app.task(name="tasks.compute_anomalies")
def compute_anomalies_task() -> Dict[str, Any]:
    """Compute anomalies in background."""
    import asyncio
    
    async def _compute():
        service = LaunchService()
        try:
            anomalies = await service.get_anomalies()
            return {
                "anomalies": anomalies,
                "count": len(anomalies),
                "status": "success"
            }
        except Exception as e:
            return {
                "error": str(e),
                "status": "error"
            }
        finally:
            if service.session and not service.session.closed:
                await service.session.close()
    
    return asyncio.run(_compute())


@celery_app.task(name="tasks.refresh_all_space_data")
def refresh_all_space_data() -> Dict[str, Any]:
    """Periodic task to refresh all space-related data."""
    import asyncio
    
    async def _refresh():
        service = LaunchService()
        results = {"tle": None, "space_weather": None, "weather": {}}
        
        try:
            # Refresh TLE data
            tle_data = await service.fetch_tle_data()
            results["tle"] = {"count": len(tle_data), "status": "success"}
            
            # Refresh space weather
            space_weather = await service.fetch_space_weather()
            results["space_weather"] = {"kp_index": space_weather.kp_index, "status": "success"}
            
            # Refresh weather for all sites
            for site in settings.LAUNCH_SITES.keys():
                try:
                    weather = await service.fetch_weather_data(site)
                    results["weather"][site] = {
                        "wind_speed": weather.wind_speed,
                        "temperature": weather.temperature,
                        "status": "success"
                    }
                except Exception as e:
                    results["weather"][site] = {"error": str(e), "status": "error"}
            
            return results
            
        except Exception as e:
            return {"error": str(e), "status": "error"}
        finally:
            if service.session and not service.session.closed:
                await service.session.close()
    
    return asyncio.run(_refresh())


# Export for easy imports
__all__ = ["celery_app", "ping", "compute_launch_windows_task", "compute_anomalies_task", "refresh_all_space_data"]
