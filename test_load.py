# test_load.py
import os

from catboost import CatBoostClassifier

model_path = "models/model.cbm"  # или твоё имя

print("Текущая папка:", os.getcwd())
print("Файл существует?", os.path.exists(model_path))

model = CatBoostClassifier()
model.load_model(model_path)
print("✅ Модель успешно загружена!")
print("Количество деревьев:", model.tree_count_)
