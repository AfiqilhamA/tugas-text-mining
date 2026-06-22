import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Konfigurasi Halaman (Biar UI-nya asik)
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

# Load Model dan Tokenizer
@st.cache_resource
def load_ml_components():
    model = load_model('model_lstm.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_ml_components()

# UI Dashboard
st.title("🧠 Text Sentiment Analysis")
st.write("Aplikasi analisis sentimen berbasis **LSTM (Long Short-Term Memory)**.")

# Input Teks dari User
user_input = st.text_area("Masukkan teks yang mau dianalisis sentimennya:", height=150)

if st.button("Analisis Sentimen 🔥"):
    if user_input.strip() == "":
        st.warning("Eits, teksnya jangan kosong bro! Isi dulu.")
    else:
        # Preprocessing teks yang masuk
        max_len = 100 # Harus sama dengan max_len saat training
        sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        
        # Prediksi pakai LSTM
        prediction = model.predict(padded_sequence)
        class_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        # Asumsi mapping label: 0 = Negatif, 1 = Positif (Sesuaikan sama data lu ya)
        # Boleh banget lu tambah logikanya kalau kelasnya ada 3 (Positif, Negatif, Netral)
        if class_idx == 1:
            sentiment = "Positif 😃"
            st.success(f"**Hasil:** Sentimen {sentiment}")
        else:
            sentiment = "Negatif 😡"
            st.error(f"**Hasil:** Sentimen {sentiment}")
            
        st.info(f"Tingkat Kepercayaan (Confidence): {confidence:.2f}%")