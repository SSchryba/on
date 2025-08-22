"""Celery application instance (production-ready stub)."""

from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "spacelaunch",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_track_started=True,
    result_expires=3600,
    timezone="UTC",
    enable_utc=True,
)

@celery_app.task(name="health.ping")
def ping() -> str:
    return "pong"
