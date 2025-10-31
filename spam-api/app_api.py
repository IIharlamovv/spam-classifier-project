from sklearn.pipeline import Pipeline
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessor import AdvancedTextPreprocessor 
import pandas as pd

import sys, types
try:
    main_mod = sys.modules.get('__main__')
    if main_mod is None or not hasattr(main_mod, '__dict__'):
        main_mod = types.ModuleType('__main__')
        sys.modules['__main__'] = main_mod
    setattr(main_mod, 'AdvancedTextPreprocessor', AdvancedTextPreprocessor)
except Exception as e:
    print(f"Compat alias for AdvancedTextPreprocessor failed: {e}")
app = FastAPI(title="Spam Classification API", version="1.0.0")

preprocessor = AdvancedTextPreprocessor()
try:
    model_data = joblib.load('spam_classifier.joblib')

    model = model_data['model']
    vectorizer = model_data['vectorizer']
    # если в файле есть "родной" препроцессор — сохраним на будущее, но пайплайн строим из (vectorizer, model)
    preproc_from_file = model_data.get('preprocessor', None)
    # единый инференс-пайплайн
    pipeline = Pipeline([('vect', vectorizer), ('clf', model)])

    # чуть-чуть логов, чтобы убедиться, что словарь не пустой
    try:
        vect = pipeline.named_steps.get('vect', None)
        if hasattr(vect, 'vocabulary_'):
            print(f"Векторизатор загружен, признаков: {len(vect.vocabulary_)}")
    except Exception:
        pass

    print("Пайплайн для инференса готов")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    print("Используем заглушку")
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
        # заглушка как была
        text_lower = message.text.lower()
        spam_keywords = ['win', 'free', 'prize', 'congratulations', 'claim', 'money', 'cash', 'award']
        spam_score = sum(1 for keyword in spam_keywords if keyword in text_lower)
        spam_prob = min(0.9, spam_score * 0.3)
        return {
            'prediction': 'spam' if spam_prob > 0.5 else 'ham',
            'spam_probability': spam_prob,
            'ham_probability': 1 - spam_prob,
            'is_spam': spam_prob > 0.5
        }

    try:
        proba = pipeline.predict_proba([message.text])[0]
        classes = list(pipeline.classes_) if hasattr(pipeline, "classes_") else list(pipeline.named_steps['clf'].classes_)
        spam_index = classes.index('spam')
        ham_index = classes.index('ham')
        pred = 'spam' if proba[spam_index] >= proba[ham_index] else 'ham'

        return {
            'prediction': pred,
            'spam_probability': float(proba[spam_index]),
            'ham_probability': float(proba[ham_index]),
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
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}