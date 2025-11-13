from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from .database import Base

# Modèle de base de données
class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    filename = Column(String)