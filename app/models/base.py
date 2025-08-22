"""Database models using SQLAlchemy."""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class LaunchWindow(Base):
    """Launch window database model."""
    __tablename__ = "launch_windows"

    id = Column(Integer, primary_key=True, index=True)
    site = Column(String, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    wind_speed = Column(Float)
    kp_index = Column(Float)
    is_viable = Column(Boolean, default=True)
    created_at = Column(DateTime)

class Anomaly(Base):
    """Space traffic anomaly database model."""
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    type = Column(String)
    description = Column(String)
    severity = Column(Integer)
    detected_at = Column(DateTime)
    resolved_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
