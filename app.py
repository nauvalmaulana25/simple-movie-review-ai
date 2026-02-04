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

# Panggil fungsinya
download_nltk_resources()

# --- Download NLTK data (hanya perlu sekali) ---
# Anda bisa menjalankan ini sekali secara manual di terminal python
# atau biarkan Streamlit menanganinya dengan @st.cache_resource
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk_data()

# --- Muat Model dan Tokenizer ---
# Gunakan st.cache_resource untuk memuat model hanya sekali
@st.cache_resource
def load_model_and_tokenizer():
    use_deep_model = os.path.exists('deep_sentiment_model.h5')

    if use_deep_model:
        print("Loading deep learning model...")
        model = tf.keras.models.load_model('deep_sentiment_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        max_len = 200
        # Mengembalikan None untuk tfidf_vectorizer agar konsisten
        return model, tokenizer, max_len, None, use_deep_model
    else:
        print("Loading traditional ML model...")
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        # Mengembalikan None untuk tokenizer dan max_len
        return model, None, None, tfidf_vectorizer, use_deep_model

model, tokenizer, max_len, tfidf_vectorizer, use_deep_model = load_model_and_tokenizer()


# --- Fungsi Preprocessing ---
def preprocess_text_ml(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def preprocess_text_dl(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# --- Antarmuka Streamlit ---
st.title("Analisis Sentimen Teks")
st.write("Masukkan ulasan atau teks di bawah ini untuk memprediksi sentimennya.")

# Text area untuk input pengguna
review = st.text_area("Teks Ulasan", "")

# Tombol untuk melakukan prediksi
if st.button("Prediksi Sentimen"):
    if review:
        if use_deep_model:
            # Preprocess untuk model deep learning
            processed_review = preprocess_text_dl(review)
            
            # Konversi ke sekuens dan lakukan padding
            sequence = tokenizer.texts_to_sequences([processed_review])
            padded_sequence = pad_sequences(sequence, maxlen=max_len)
            
            # Lakukan prediksi
            prediction = model.predict(padded_sequence)[0][0]
            sentiment = "Positif" if prediction > 0.5 else "Negatif"
            confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
            
        else:
            # Preprocess untuk model ML tradisional
            processed_review = preprocess_text_ml(review)
            
            # Transformasi menggunakan TF-IDF
            review_tfidf = tfidf_vectorizer.transform([processed_review])
            
            # Lakukan prediksi
            prediction_val = model.predict(review_tfidf)[0]
            sentiment = "Positif" if prediction_val == 1 else "Negatif"
            
            # Dapatkan probabilitas untuk confidence
            proba = model.predict_proba(review_tfidf)[0]
            confidence = float(proba[1]) if prediction_val == 1 else float(proba[0])
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        if sentiment == "Positif":
            st.success(f"Sentimen: {sentiment}")
        else:
            st.error(f"Sentimen: {sentiment}")
        
        st.write(f"Tingkat Keyakinan: {confidence:.2%}")

    else:

        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")

