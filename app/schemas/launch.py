"""Pydantic schemas for data validation."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class LaunchSite(BaseModel):
    """Launch site schema."""
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    
    class Config:
        from_attributes = True

class WeatherCondition(BaseModel):
    """Weather condition schema."""
    wind_speed: float
    wind_direction: float
    temperature: float
    pressure: float
    
    class Config:
        from_attributes = True

class SpaceWeather(BaseModel):
    """Space weather schema."""
    kp_index: float
    solar_wind_speed: float
    magnetic_field_bt: float
    timestamp: datetime
    
    class Config:
        from_attributes = True

class LaunchWindow(BaseModel):
    """Launch window schema."""
    start_time: datetime
    end_time: datetime
    site: str
    weather: WeatherCondition
    space_weather: SpaceWeather
    is_viable: bool
    constraints: List[str]
    
    class Config:
        from_attributes = True

class Anomaly(BaseModel):
    """Anomaly schema."""
    id: int
    type: str
    description: str
    severity: int
    detected_at: datetime
    resolved_at: Optional[datetime] = None
    is_active: bool
    
    class Config:
        from_attributes = True
