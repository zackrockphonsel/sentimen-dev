import streamlit as st
import pandas as pd
import joblib
import time
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(
    page_title="Analisis - TF-IDF",
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

st.title("Ekstraksi Fitur (TF-IDF)")

# Fungsi untuk melakukan ekstraksi fitur TF-IDF
def extract_features(data):
    # Menggabungkan teks menjadi kalimat-kalimat
    documents = [" ".join(tokens) for tokens in data['final_prep']]
    
    # Membuat objek TF-IDF Vectorizer
    cv = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    # Memulai perhitungan waktu
    start_time = time.time()

    # Melakukan ekstraksi fitur TF-IDF
    count = cv.fit_transform(documents)
    tfidf_features = tfidf_vectorizer.fit_transform(documents)

    # Mengakhiri perhitungan waktu dan mencatat durasi
    end_time = time.time()
    execution_time = end_time - start_time
    execution_time_ekstraksi_fitur = end_time - start_time
    st.success(f'Done! Waktu Eksekusi: {execution_time_ekstraksi_fitur:.2f} detik')
    
    return tfidf_features, tfidf_vectorizer

# Memeriksa apakah data telah diproses sebelumnya
if 'labeled_data' in st.session_state:
    labeled_data = st.session_state['labeled_data']
    # Melakukan proses ekstraksi fitur TF-IDF
    tfidf_features, tfidf_vectorizer = extract_features(labeled_data)

    # Menyimpan hasil ekstraksi fitur TF-IDF dan vektorizer TF-IDF ke dalam session state
    st.session_state['tfidf_features'] = tfidf_features
    st.session_state['tfidf_vectorizer'] = tfidf_vectorizer

    # Menampilkan hasil ekstraksi fitur
    st.write("Daftar Fitur:", tfidf_vectorizer.get_feature_names_out())
    st.write("Shape:", tfidf_features.shape)

    # Mendapatkan daftar fitur (kata-kata) dari TF-IDF Vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Membuat DataFrame untuk menampilkan matriks TF-IDF dengan keterangan kata dan nilai TF-IDF
    tfidf_df = pd.DataFrame(data=tfidf_features.toarray(), columns=feature_names)
    st.write(tfidf_df)

    st.write("Matriks TF-IDF:", tfidf_features.toarray())

    # Simpan model TF-IDF ke dalam folder model
    joblib.dump(tfidf_vectorizer, 'model/tfidf_vectorizer.pkl')

    # st.success("Model TF-IDF telah dilatih dan disimpan.")

else:
    st.error("Data labeling belum dilakukan. Silakan lakukan proses labeling terlebih dahulu di halaman Labelling.")
