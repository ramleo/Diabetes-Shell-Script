#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib, pandas as pd

_BASE = Path(__file__).parent.resolve()

app = FastAPI(title="ML Prediction API")

try:
    pipeline = joblib.load(_BASE / "models" / "final_pipeline.pkl")
except FileNotFoundError:
    raise RuntimeError("models/final_pipeline.pkl not found. Train the pipeline first.")
try:
    label_encoder = joblib.load(_BASE / "models" / "label_encoder.pkl")
except FileNotFoundError:
    raise RuntimeError("models/label_encoder.pkl not found.")

class InputData(BaseModel):
    Pregnancies: Optional[float] = None
    Glucose: Optional[float] = None
    BloodPressure: Optional[float] = None
    SkinThickness: Optional[float] = None
    Insulin: Optional[float] = None
    BMI: Optional[float] = None
    DiabetesPedigreeFunction: Optional[float] = None
    Age: Optional[float] = None

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([{"Pregnancies": data.Pregnancies, "Glucose": data.Glucose, "BloodPressure": data.BloodPressure, "SkinThickness": data.SkinThickness, "Insulin": data.Insulin, "BMI": data.BMI, "DiabetesPedigreeFunction": data.DiabetesPedigreeFunction, "Age": data.Age}])
    pred = pipeline.predict(df)[0]
    label = label_encoder.inverse_transform([pred])[0]
    proba = pipeline.predict_proba(df)[0].tolist()
    return {"prediction": str(label), "probabilities": proba}

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.DataFrame([{"Pregnancies": d.Pregnancies, "Glucose": d.Glucose, "BloodPressure": d.BloodPressure, "SkinThickness": d.SkinThickness, "Insulin": d.Insulin, "BMI": d.BMI, "DiabetesPedigreeFunction": d.DiabetesPedigreeFunction, "Age": d.Age} for d in data])
    preds = pipeline.predict(df)
    labels = label_encoder.inverse_transform(preds).tolist()
    return {"predictions": [str(l) for l in labels]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
