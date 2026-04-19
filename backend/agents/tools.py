import joblib
import numpy as np
import json
from backend.models.schemas import PatientFeatures
from backend.database import PatientRecord, SessionLocal

# Load ML models (assuming running from root)
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    model = None
    scaler = None

feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def predict_tool(features: PatientFeatures) -> dict:
    if model is None or scaler is None:
        return {"risk_probability": 0.5, "label": "Model Not Found", "scaled_features": [0]*13}
    feature_arr = np.array([[getattr(features, f) for f in feature_names]])
    scaled = scaler.transform(feature_arr)
    prob = model.predict_proba(scaled)[0][1]
    label = "High Risk" if prob >= 0.5 else "Low Risk"
    return {"risk_probability": float(prob), "label": label, "scaled_features": scaled[0].tolist()}

def explain_tool(scaled_features: list) -> list:
    if model is None:
        return []
    weights = model.coef_[0]
    contributions = []
    for i, w in enumerate(weights):
        val = w * scaled_features[i]
        contributions.append({"feature": feature_names[i], "contribution": float(val)})
    
    # Sort by absolute contribution to find highest impact
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    top_3 = contributions[:3]
    return top_3

def memory_tool(risk_score: float, top_factors: list) -> int:
    db = SessionLocal()
    record = PatientRecord(
        risk_score=risk_score,
        top_factors=json.dumps(top_factors)
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    record_id = record.id
    db.close()
    return record_id

def get_history():
    db = SessionLocal()
    records = db.query(PatientRecord).order_by(PatientRecord.timestamp.asc()).all()
    # just return list of dicts
    out = [{"id": r.id, "timestamp": r.timestamp.isoformat(), "risk_score": r.risk_score, "top_factors": json.loads(r.top_factors)} for r in records]
    db.close()
    return out
