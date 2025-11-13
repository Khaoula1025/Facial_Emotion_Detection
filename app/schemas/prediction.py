from pydantic import BaseModel

class storePrediction(BaseModel):
    emotion : str
    confidence : float
    filename : str