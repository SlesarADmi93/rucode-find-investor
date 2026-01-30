import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
seed = config["seed"]
n_samples = config["n_samples"]


def generate_synthetic_data(n_samples=n_samples, seed=seed):
    np.random.seed(seed)

    # 1. Генерируем базовые числовые признаки + целевую переменную
    # Используем make_classification для создания зависимости
    # между фичами и target
    X_base, y = make_classification(
        n_samples=n_samples,
        n_features=4,  # age, balance, offer_amount, previous_investments
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=seed,
    )

    # Приводим к более "человеческим" шкалам
    age = np.clip(
        (X_base[:, 0] * 15 + 40).astype(int), 18, 80
    )  # возраст от 18 до 80
    balance = np.exp(
        X_base[:, 1] * 1.5 + 10
    )  # логнормальное распределение баланса
    offer_amount = np.clip(
        (X_base[:, 2] * 500 + 1000).astype(int), 100, 5000
    )  # предложения от 100 до 5000
    previous_investments = np.clip(
        (X_base[:, 3] * 3 + 5).astype(int), 0, 20
    )  # количество прошлых инвестиций

    risk_profile = np.array(
        [
            np.random.choice(
                ["low", "medium", "high"],
                p=[0.5, 0.3, 0.2] if yi == 1 else [0.2, 0.3, 0.5],
            )
            for yi in y
        ]
    )

    marketing_channel = np.array(
        [
            np.random.choice(
                ["email", "social", "search", "direct"],
                p=[0.4, 0.2, 0.2, 0.2] if yi == 1 else [0.2, 0.3, 0.3, 0.2],
            )
            for yi in y
        ]
    )

    membership_tier = np.array(
        [
            np.random.choice(
                ["bronze", "silver", "gold"],
                p=[0.3, 0.4, 0.3] if yi == 1 else [0.6, 0.3, 0.1],
            )
            for yi in y
        ]
    )

    responded_before = np.array(
        [
            np.random.choice([0, 1], p=[0.3, 0.7] if yi == 1 else [0.8, 0.2])
            for yi in y
        ]
    )

    # 3. Собираем DataFrame
    df = pd.DataFrame(
        {
            "customer_id": range(1, n_samples + 1),
            "age": age,
            "balance": balance,
            "risk_profile": risk_profile,
            "marketing_channel": marketing_channel,
            "offer_amount": offer_amount,
            "previous_investments": previous_investments,
            "responded_before": responded_before,
            "membership_tier": membership_tier,
            "accepted": y,
        }
    )

    return df
