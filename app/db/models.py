from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

# class EmotionPrediction(Base):
#     __tablename__ = "predictions"
#     id = Column(Integer, primary_key=True)
#     emotion = Column(String, nullable=False)
#     confidence = Column(Float, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
