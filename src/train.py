# src/train.py
import os

from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import yaml

from .data import generate_synthetic_data


def train_model():
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Генерация данных
    df = generate_synthetic_data(seed=config["seed"])
    test_df = generate_synthetic_data(seed=config["seed"] + 1)

    # Разделение
    X = df.drop(columns=["customer_id", "accepted"])
    y = df["accepted"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config["seed"], stratify=y
    )

    # Обучение
    model = CatBoostClassifier(
        cat_features=config["cat_features"],
        random_seed=config["seed"],
        verbose=100,  # noqa: E501
    )
    model.fit(X_train, y_train)

    # Разделение на признаки и целевую переменную
    X_test = test_df.drop(columns=["customer_id", "accepted"])
    y_test = test_df["accepted"]

    # Оценка
    val_preds = model.predict(X_val)
    f1 = f1_score(y_val, val_preds)
    print(f"F1 Score на валидации: {f1:.6f}")

    # Оценка на тесте
    test_preds = model.predict(X_test)
    test_f1 = f1_score(y_test, test_preds)
    print(f"F1 Score на hold-out тесте: {test_f1:.6f}")

    # Сохранение
    os.makedirs(config["model_dir"], exist_ok=True)
    model.save_model(os.path.join(config["model_dir"], config["model_name"]))

    # Сохранение данных для инференса
    os.makedirs(config["data_dir"], exist_ok=True)

    return model, f1
