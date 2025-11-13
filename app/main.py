from fastapi import FastAPI,File, UploadFile,Depends
from fastapi.responses import JSONResponse
from app.db.database import Base ,engine 
from app.db.models import Prediction
from app.schemas.prediction import storePrediction 
from app.utils.predictionScript import predict_emotions
from app.db.database import SessionLocal
from sqlalchemy.orm import Session
app = FastAPI(title="Emotion Detection API")
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get('/')
def start():
    return {'message':'Bonjour'}

@app.post('/predict_emotion')
async def emotion_detector(
    file:UploadFile=File(...),
    db: Session = Depends(get_db)):
    image= await file.read()
    resultat=predict_emotions(image)
    prediction=Prediction( emotion=resultat['emotion'] ,
                          confidence=resultat['confidence'], 
                          filename=file.filename)
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return JSONResponse({
     'emotion': resultat['emotion'],
     'score': round(resultat['confidence'], 3)
    })
    

@app.get('/history')
def get_record( db: Session = Depends(get_db)):
    return db.query(Prediction).all()