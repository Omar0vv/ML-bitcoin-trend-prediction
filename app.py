from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import joblib
import logging
import os

from config import *


app = FastAPI(title="Bitcoin Trend Prediction System", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)
metrics = joblib.load(METRICS_PATH)


@app.get("/")
def home(request: Request):
    context = {
        "prediction": None,
        "accuracy": round(metrics["accuracy"], 4),
        "f1_score": round(metrics["f1_score"], 4),
        "btc_price": "Demo Mode"
    }

    return templates.TemplateResponse(
        request,
        "index.html",
        context
    )


@app.post("/")
def predict(
    request: Request,
    Open: float = Form(...),
    High: float = Form(...),
    Low: float = Form(...),
    Close: float = Form(...),
    Volume: float = Form(...),
    Daily_Return: float = Form(...),
    MA_7: float = Form(...),
    MA_14: float = Form(...),
    MA_30: float = Form(...),
    Volatility_7: float = Form(...)
):
    input_data = pd.DataFrame([{
        "Open": Open,
        "High": High,
        "Low": Low,
        "Close": Close,
        "Volume": Volume,
        "Daily_Return": Daily_Return,
        "MA_7": MA_7,
        "MA_14": MA_14,
        "MA_30": MA_30,
        "Volatility_7": Volatility_7
    }])

    input_data = input_data[features]

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    result = "UP" if prediction == 1 else "DOWN"

    explanation = (
        "Bitcoin price may increase tomorrow."
        if result == "UP"
        else "Bitcoin price may decrease tomorrow."
    )

    logging.info(f"Prediction: {result}")

    context = {
        "prediction": result,
        "confidence_down": round(float(probabilities[0]), 4),
        "confidence_up": round(float(probabilities[1]), 4),
        "explanation": explanation,
        "accuracy": round(metrics["accuracy"], 4),
        "f1_score": round(metrics["f1_score"], 4),
        "btc_price": "Demo Mode"
    }

    return templates.TemplateResponse(
        request,
        "index.html",
        context
    )