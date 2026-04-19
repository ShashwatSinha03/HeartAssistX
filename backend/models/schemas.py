from pydantic import BaseModel
from typing import List

class PatientFeatures(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

class PatientRecordResponse(BaseModel):
    id: int
    timestamp: str
    risk_score: float
    top_factors: str
