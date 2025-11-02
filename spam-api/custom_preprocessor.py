# custom_preprocessor.py
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

# Гарантируем наличие ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Кастомный трансформер для препроцессинга текста"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Очистка текста"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_stem(self, text):
        """Токенизация и стемминг"""
        tokens = word_tokenize(text)
        filtered = [
            self.stemmer.stem(w)
            for w in tokens
            if w not in self.stop_words and len(w) > 2
        ]
        return filtered
    
    def preprocess_text(self, text):
        """Полный препроцессинг текста"""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        tokens = self.tokenize_and_stem(cleaned)
        return " ".join(tokens)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Трансформация для sklearn Pipeline"""
        return [self.preprocess_text(text) for text in X]