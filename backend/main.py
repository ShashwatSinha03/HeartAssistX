from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models.schemas import PatientFeatures
from backend.agents.tools import predict_tool, get_history
from backend.agents.groq_agent import call_agent

import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="HeartAssistX Dual-Version Base")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/predict")
async def v1_predict(features: PatientFeatures):
    res = predict_tool(features)
    return {"risk_probability": res["risk_probability"], "label": res["label"]}

@app.post("/v2/analyze")
async def v2_analyze(features: PatientFeatures):
    # Pass plain dict to agent
    result = call_agent(features.model_dump())
    return result

@app.get("/v2/history")
async def v2_history():
    return get_history()

