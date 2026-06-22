import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

from xgboost import train

# 1. Load Dataset (pakai link RAW biar bisa langsung dibaca pandas)
url = "https://raw.githubusercontent.com/rzyunanda/Text-Mining-Session-12/main/data.csv"
df = pd.read_csv(url)

# NOTES PENTING: Cek dulu nama kolom di datasetnya apa!
# Gue asumsikan nama kolom teksnya 'text' dan labelnya 'label'
# Kalau beda, tinggal ganti string di bawah ini sesuai nama kolom di CSV-nya
text_col = 'text' 
label_col = 'label'

# Drop missing values kalau ada
df = df.dropna(subset=[text_col, label_col])

# Bikin label jadi angka (misal: 0 untuk negatif, 1 untuk positif)
# Kalau labelnya udah angka dari sananya, skip aja bagian ini.
if df[label_col].dtype == type(object):
    df['target'] = df[label_col].astype('category').cat.codes
else:
    df['target'] = df[label_col]

X = df[text_col].values
y = df['target'].values

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Tokenisasi & Padding (Biar teks jadi angka yang dimengerti LSTM)
max_vocab = 10000
max_len = 100

tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len, padding='post')
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len, padding='post')

# 3. Build Model LSTM
model = Sequential([
    Embedding(input_dim=max_vocab, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax') # Softmax untuk klasifikasi multi-class / sigmoid untuk biner
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train Model
print("Training model LSTM... Gass!")
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_data=(X_test_pad, y_test))

# 5. Save Model dan Tokenizer (Ini yang bakal dipakai di Streamlit)
model.save('model_lstm.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Berhasil! model_lstm.h5 dan tokenizer.pickle udah jadi.")