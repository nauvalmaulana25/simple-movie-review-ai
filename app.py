import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

st.set_page_config(
    page_title="Analyze Movie Review AI",
    page_icon="🎬",
    layout="centered"
)

# DOWNLOAD NLTK RESOURCES
@st.cache_resource
def setup_nltk():
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for res in resources:
        nltk.download(res)

setup_nltk()

# LOAD MODEL & TOKENIZER
@st.cache_resource
def load_assets():
    # MODEL CHECKING
    use_deep_model = os.path.exists('deep_sentiment_model.h5')
    
    model = None
    tokenizer = None
    tfidf_vectorizer = None
    max_len = 200

    if use_deep_model:
        model = tf.keras.models.load_model('deep_sentiment_model.h5')
        if os.path.exists('tokenizer.pkl'):
            with open('tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
    else:
        if os.path.exists('sentiment_model.pkl'):
            with open('sentiment_model.pkl', 'rb') as f:
                model = pickle.load(f)
        if os.path.exists('tfidf_vectorizer.pkl'):
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
                
    return model, tokenizer, tfidf_vectorizer, use_deep_model, max_len

model, tokenizer, tfidf_vectorizer, use_deep_model, max_len = load_assets()

# PREPROCESSING FUNCTION
def preprocess_text(text, method='dl'):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    
    if method == 'ml':
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    else:
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
    return ' '.join(tokens)

# STREAMLIT UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="📊")

st.title("📊 AI Movie Sentiment Analysis")
st.write(f"Model Detected: **{'Deep Learning (.h5)' if use_deep_model else 'Machine Learning (.pkl)'}**")

review = st.text_area("Input any movie review in English:", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("please enter a movie review to analyze.")
    elif model is None:
        st.error("Model not found. Please ensure the model files are in place.")
    else:
        with st.spinner('Sedang menganalisis...'):
            if use_deep_model:
                # Deep Learning
                processed = preprocess_text(review, method='dl')
                sequence = tokenizer.texts_to_sequences([processed])
                padded = pad_sequences(sequence, maxlen=max_len)
                
                prediction = model.predict(padded)[0][0]
                sentiment = "Positive" if prediction > 0.5 else "Negative"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            else:
                # ML Traditional
                processed = preprocess_text(review, method='ml')
                vector = tfidf_vectorizer.transform([processed])
                
                prediction = model.predict(vector)[0]
                
                if prediction == 1 or str(prediction).lower() == 'positive':
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"
                
                # Probability if any
                try:
                    proba = model.predict_proba(vector)[0]
                    confidence = max(proba)
                except:
                    confidence = None

            # Show Result
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == "Positive":
                    st.success(f"### Sentiment: {sentiment} 😊")
                else:
                    st.error(f"### Sentiment: {sentiment} ☹️")
            
            with col2:
                if confidence is not None:
                    st.metric("Confidence Score", f"{confidence:.2%}")



