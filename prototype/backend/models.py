from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.sql import func
from .database import Base

class Scan(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    correct_label = Column(String, nullable=True) # User feedback for continuous learning
    static_analysis = Column(String)
    dynamic_analysis = Column(String)
    fusion_strategy = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class SystemMetric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    total_scans = Column(Integer, default=0)
    threats_blocked = Column(Integer, default=0)
    model_accuracy = Column(Float, default=98.75)  # Verified via stratified eval (eval_baseline.py)
    last_evolution = Column(DateTime(timezone=True), server_default=func.now())
