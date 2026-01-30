# Используем актуальный Python 3.12
FROM python:3.12-slim

# Устанавливаем зависимости ОС (если нужны компиляторы для некоторых pip-пакетов)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем зависимости отдельно — для кэширования слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Открываем порт FastAPI
EXPOSE 8000

# Запуск 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]