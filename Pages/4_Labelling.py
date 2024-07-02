import streamlit as st
import os
import pandas as pd
import datetime
import time
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Analisis - Labelling",
    page_icon="⚙",
    layout="wide",
)

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# Function to load lexicon from MySQL database
def load_lexicon():
    # Koneksi ke database MySQL menggunakan SQLAlchemy
    engine = create_engine('mysql+pymysql://root:@localhost/db_skripsi')
    
    # Query untuk mengambil kamus positif dari tabel lexicon_positif
    query_positive = "SELECT kata, skor FROM lex_positive"  # Perbaikan nama tabel lexicon_positif
    df_positive = pd.read_sql(query_positive, engine)
    
    # Query untuk mengambil kamus negatif dari tabel lexicon_negatif
    query_negative = "SELECT kata, skor FROM lex_negative"  # Perbaikan nama tabel lexicon_negatif
    df_negative = pd.read_sql(query_negative, engine)
    
    # Konversi DataFrame ke dictionary
    lexicon_positive = pd.Series(df_positive.skor.values, index=df_positive.kata).to_dict()
    lexicon_negative = pd.Series(df_negative.skor.values, index=df_negative.kata).to_dict()
    
    return lexicon_positive, lexicon_negative

# Load lexicons from MySQL database
lexicon_positive, lexicon_negative = load_lexicon()

# Function to determine sentiment polarity
def lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        if word in lexicon_negative:
            score += lexicon_negative[word]
    
    sentiment = 'positif' if score >= 0 else 'negatif'
    return score, sentiment

# Labeling
st.header("Labeling")
if 'preprocessed_data' in st.session_state:
    data = st.session_state['preprocessed_data']
    
    # Melakukan pelabelan dengan memulai perhitungan waktu
    start_time = time.time()

    # Melakukan pelabelan
    results = data['final_prep'].apply(lexicon_indonesia)
    results = list(zip(*results))
    data['sentimen_skor'] = results[0]
    data['sentimen'] = results[1]

    # Mengakhiri perhitungan waktu dan mencatat durasi
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_labelling = end_time - start_time
    st.success(f'Done! Waktu Eksekusi: {execution_time_labelling:.2f} detik')
    
    st.dataframe(data['sentimen'].value_counts(), width=1000)
    st.dataframe(data[['final_prep', 'sentimen_skor', 'sentimen']].head(20), width=1000)

    # Button to save labelling result
    if st.button("Simpan Hasil Labelling"):
        result_file_path = os.path.join("hasil", "result_labelling.csv")
        data.to_csv(result_file_path, index=False)
        st.success(f"Hasil labelling telah disimpan: {result_file_path}")
        
else:
    st.error("Harap jalankan proses preprocessing terlebih dahulu melalui menu 'Preprocessing'.")

# Pastikan data telah didefinisikan sebelum menyimpan ke session state
if 'data' in locals() or 'data' in globals():
    st.session_state['labeled_data'] = data

