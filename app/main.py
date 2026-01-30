# app/main.py
from catboost import CatBoostClassifier
import joblib
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Загрузка
model = CatBoostClassifier()
model.load_model("models/model.cbm")
FEATURE_NAMES = [
    "age",
    "balance",
    "risk_profile",
    "marketing_channel",
    "offer_amount",
    "previous_investments",
    "responded_before",
    "membership_tier",
]


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ⚠️ ВАЖНО: названия аргументов ДОЛЖНЫ совпадать с FEATURE_NAMES!
@app.post("/predict")
def predict(
    age: int = Form(...),
    balance: float = Form(...),
    risk_profile: str = Form(...),
    marketing_channel: str = Form(...),
    offer_amount: float = Form(...),
    previous_investments: int = Form(...),
    responded_before: bool = Form(False),
    membership_tier: str = Form(...),
):
    # Собираем словарь с теми же ключами, что и в FEATURE_NAMES
    data = {
        "age": [age],
        "balance": [balance],
        "risk_profile": [risk_profile],
        "marketing_channel": [marketing_channel],
        "offer_amount": [offer_amount],
        "previous_investments": [previous_investments],
        "responded_before": [int(responded_before)],
        "membership_tier": [membership_tier],
    }

    # Создаём DataFrame в правильном порядке
    df = pd.DataFrame(data)[FEATURE_NAMES]

    # Предсказание
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    return {"churn": bool(pred), "probability": round(float(proba), 4)}
