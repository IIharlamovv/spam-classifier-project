from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Dict, Any

# Импортируем кастомный трансформер
from custom_preprocessor import TextPreprocessor

app = FastAPI(title="Spam Classification API", version="1.0.0")

# Загружаем единый пайплайн
try:
    pipeline = joblib.load('spam_classifier.joblib')
    print("Модель успешно загружена")
    print(f"Шаги пайплайна: {list(pipeline.named_steps.keys())}")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    pipeline = None

class Message(BaseModel):
    text: str

class PredictionResult(BaseModel):
    prediction: str
    spam_probability: float
    ham_probability: float
    is_spam: bool

@app.get("/")
async def root():
    return {"message": "Spam Classification API", "status": "running"}

@app.post("/predict", response_model=PredictionResult)
async def predict(message: Message):
    if pipeline is None:
        return {
            'prediction': 'error',
            'spam_probability': 0.5,
            'ham_probability': 0.5,
            'is_spam': False
        }

    try:
        # Прямой вызов пайплайна
        proba = pipeline.predict_proba([message.text])[0]
        pred = pipeline.predict([message.text])[0]
        
        # Получаем индексы классов
        classes = list(pipeline.named_steps['classifier'].classes_)
        spam_idx = classes.index('spam')
        ham_idx = classes.index('ham')
        
        return {
            'prediction': pred,
            'spam_probability': float(proba[spam_idx]),
            'ham_probability': float(proba[ham_idx]),
            'is_spam': pred == 'spam'
        }
    except Exception as e:
        return {
            "prediction": "error",
            "spam_probability": 0.5,
            "ham_probability": 0.5,
            "is_spam": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if pipeline is not None else "unhealthy", 
        "model_loaded": pipeline is not None
    }