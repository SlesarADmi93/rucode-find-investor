from catboost import CatBoostClassifier
import pandas as pd
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Загрузка модели
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

# Белый список допустимых значений для демо-режима
DEMO_ALLOWED_VALUES = {
    'age': [25, 30, 35, 40, 45, 50],
    'balance': [10000.0, 25000.0, 50000.0, 75000.0, 100000.0],
    'offer_amount': [100.0, 500.0, 1000.0, 2000.0],
    'previous_investments': [0, 1, 2, 3, 5]
}

DEMO_ALLOWED_CATEGORIES = {
    'risk_profile': ['Low', 'Medium', 'High'],
    'marketing_channel': ['Email', 'Social', 'Referral', 'Banner'],
    'membership_tier': ['Basic', 'Gold', 'Platinum']
}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Возвращает главную страницу с формой ввода
    return templates.TemplateResponse("index.html", {"request": request})

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
    demo_mode: str = Form(None)
):
    # Проверка флага демо-режима
    if demo_mode != 'true':
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'Только демо-режим разрешён в учебном проекте',
                'hint': 'Добавьте demo_mode=true в форму'
            }
        )
    
    # Валидация числовых параметров
    if not is_demo_data(age, balance, offer_amount, previous_investments):
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'Обнаружены неразрешённые значения',
                'hint': 'Используйте только демо-данные из списка',
                'allowed_values': DEMO_ALLOWED_VALUES
            }
        )
    
    # Валидация категориальных параметров
    if not is_demo_categories(risk_profile, marketing_channel, membership_tier):
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'Обнаружены неразрешённые категориальные значения',
                'hint': 'Используйте только демо-категории из списка',
                'allowed_categories': DEMO_ALLOWED_CATEGORIES
            }
        )
    
    # Формирование данных для предсказания
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

    # Создание DataFrame в строгом порядке признаков
    df = pd.DataFrame(data)[FEATURE_NAMES]

    # Выполнение предсказания модели
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    # Формирование ответа клиенту
    return {
        "success": True,
        "churn": bool(pred),
        "probability": round(float(proba), 4),
        "disclaimer": "Данные не сохранены. Учебный проект."
    }

def is_demo_data(age: int, balance: float, offer_amount: float, previous_investments: int) -> bool:
    # Проверяет соответствие числовых значений белому списку
    return (
        age in DEMO_ALLOWED_VALUES['age'] and
        balance in DEMO_ALLOWED_VALUES['balance'] and
        offer_amount in DEMO_ALLOWED_VALUES['offer_amount'] and
        previous_investments in DEMO_ALLOWED_VALUES['previous_investments']
    )

def is_demo_categories(
    risk_profile: str, 
    marketing_channel: str, 
    membership_tier: str
) -> bool:
    # Проверяет соответствие категориальных значений белому списку
    return (
        risk_profile in DEMO_ALLOWED_CATEGORIES['risk_profile'] and
        marketing_channel in DEMO_ALLOWED_CATEGORIES['marketing_channel'] and
        membership_tier in DEMO_ALLOWED_CATEGORIES['membership_tier']
    )