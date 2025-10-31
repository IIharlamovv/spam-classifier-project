FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Затем копируем остальной код
COPY . .

CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "8000"]