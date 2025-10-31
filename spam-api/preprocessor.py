import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Скачиваем данные при импорте
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AdvancedTextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_stem(self, text):
        tokens = word_tokenize(text)
        filtered_tokens = [
            self.stemmer.stem(word) 
            for word in tokens 
            if word not in self.stop_words and len(word) > 2
        ]
        return filtered_tokens
        
    def preprocess(self, text):
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return ""
        tokens = self.tokenize_and_stem(cleaned_text)
        return ' '.join(tokens)