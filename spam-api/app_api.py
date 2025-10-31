from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessor import AdvancedTextPreprocessor
import pandas as pd

app = FastAPI(title="Spam Classification API", version="1.0.0")

# Создаем новый препроцессор (совместимый с тем, что в модели)
preprocessor = AdvancedTextPreprocessor()

# Пытаемся загрузить модель
try:
    model_data = joblib.load('spam_classifier.joblib')
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    # Используем наш новый препроцессор вместо старого
    print("Модель и векторизатор загружены")
    print("Используем новый препроцессор")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    print("Используем заглушку")
    model = None
    vectorizer = None

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
    if model is None:
        # Умная заглушка
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
        # Используем НОВЫЙ препроцессор
        processed_text = preprocessor.preprocess(message.text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        
        # Определяем индексы для спама и не спама
        spam_index = list(model.classes_).index('spam')
        ham_index = list(model.classes_).index('ham')
        
        result = {
            'prediction': prediction,
            'spam_probability': float(probability[spam_index]),
            'ham_probability': float(probability[ham_index]),
            'is_spam': prediction == 'spam'
        }
        
        return result
    
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