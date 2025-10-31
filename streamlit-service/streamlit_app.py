import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Настройки страницы
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="wide"
)

# Заголовок приложения
st.title("📧 Spam Message Classifier")
st.markdown("Определение спам-сообщений с помощью ML модели")

# URL API (будет меняться в зависимости от окружения)
# API_URL = "http://localhost:8000"  # Для локальной разработки
API_URL = "http://spam-api:8000"  # Для Docker

def predict_spam(message):
    """Функция для отправки запроса к API"""
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": message})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# Основной интерфейс
tab1, tab2, tab3 = st.tabs(["🔍 Классификатор", "📊 Анализ", "ℹ️ О проекте"])

with tab1:
    st.header("Проверка сообщений")
    
    # Ввод сообщения
    message = st.text_area(
        "Введите текст сообщения для проверки:",
        placeholder="Например: Congratulations! You won a $1000 prize...",
        height=100
    )
    
    # Кнопка предсказания
    if st.button("Проверить на спам", type="primary"):
        if message.strip():
            with st.spinner("Анализируем сообщение..."):
                result = predict_spam(message)
                
            if "error" not in result:
                # Отображение результатов
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result["is_spam"]:
                        st.error("🚨 СПАМ")
                    else:
                        st.success("✅ НЕ СПАМ")
                
                with col2:
                    st.metric(
                        label="Вероятность спама", 
                        value=f"{result['spam_probability']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        label="Вероятность не спама", 
                        value=f"{result['ham_probability']:.2%}"
                    )
                
                # Визуализация вероятностей
                fig = px.bar(
                    x=["Не спам", "Спам"],
                    y=[result["ham_probability"], result["spam_probability"]],
                    labels={"x": "Класс", "y": "Вероятность"},
                    color=["Не спам", "Спам"],
                    color_discrete_map={"Не спам": "green", "Спам": "red"}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Ошибка: {result['error']}")
        else:
            st.warning("Пожалуйста, введите текст сообщения")

with tab2:
    st.header("Статистика и анализ")
    
    # Примеры для тестирования
    st.subheader("Тестовые примеры")
    
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, are we still meeting for lunch tomorrow?",
        "Congratulations! You've won a $1000 gift card. Click here to claim.",
        "Ok, see you later then",
        "URGENT: Your bank account has been suspended. Verify your details now."
    ]
    
    results = []
    for msg in test_messages:
        result = predict_spam(msg)
        if "error" not in result:
            results.append({
                "Сообщение": msg[:50] + "..." if len(msg) > 50 else msg,
                "Результат": "Спам" if result["is_spam"] else "Не спам",
                "Вер. спама": f"{result['spam_probability']:.2%}",
                "Вер. не спама": f"{result['ham_probability']:.2%}"
            })
    
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

with tab3:
    st.header("О проекте")
    
    st.markdown("""
    ### Spam Classification Project
    
    Этот проект использует машинное обучение для классификации текстовых сообщений на спам и не спам.
    
    **Технологии:**
    - FastAPI для REST API
    - Streamlit для веб-интерфейса
    - Scikit-learn для ML модели
    - Docker для контейнеризации
    
    **Модель:**
    - Алгоритм: SVC
    - Векторизация: TF-IDF
    - Точность: >98%
    
    **API endpoints:**
    - `POST /predict` - классификация сообщения
    - `GET /health` - проверка статуса API
    """)

# Футер
st.markdown("---")
st.markdown("Spam Classifier Project © 2025")