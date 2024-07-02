import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

st.set_page_config(
    page_title="Analisis - Klasifikasi",
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

st.title("Klasifikasi")

# Check if TF-IDF features are available
if 'tfidf_features' in st.session_state:
    tfidf_features = st.session_state['tfidf_features']
    
    # Pisahkan kolom fitur dan target
    X = tfidf_features  # Gunakan hasil ekstraksi fitur TF-IDF sebagai fitur
    y = st.session_state['labeled_data']['sentimen']  # Kolom target
    indices = st.session_state['labeled_data'].index  # Ambil indeks DataFrame

    # Get user choice for train-test split ratio
    train_test_ratio_options = {'75 data train 25 data testing': 0.75,
                                '70 data train 30 data testing': 0.7,
                                '80 data train 20 data testing': 0.8,
                                '90 data train 10 data testing': 0.9}
    selected_ratio = st.radio("Pilih Proporsi Data Train dan Test", options=list(train_test_ratio_options.keys()))
    train_test_ratio = train_test_ratio_options[selected_ratio]
    st.write(f"Proporsi Data Train: {train_test_ratio*100}%, Proporsi Data Test: {(1-train_test_ratio)*100}%")

    # Split data train dan data testing
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=1-train_test_ratio, random_state=42)

    # # Split data train dan data testing (80:20)
    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y, indices, test_size=0.2, random_state=42)
    
    # Tampilkan jumlah data training dan data testing
    st.write(f"Jumlah Data Train: {X_train.shape[0]}, Jumlah Data Test: {X_test.shape[0]}")

    # Latih model Naive Bayes
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train, y_train)

    # Simpan model Naive Bayes
    model_path = 'model/naive_bayes_model.pkl'
    joblib.dump(naive_bayes_model, model_path)
    st.success("Model Naive Bayes telah dilatih dan disimpan.")

    # # Display classification report Data Train
    # st.subheader("Classification Report Data Training")
    # y_pred = naive_bayes_model.predict(X_train)
    # report = classification_report(y_train, y_pred)
    # st.text(report)
    
    # Display classification report
    st.subheader("Classification Report Data Testing")  
    y_pred = naive_bayes_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    st.text(report)

    # Display confusion matrix
    st.subheader("Confusion Matrix Data Testing")
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_names = np.unique(y)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Data Prediksi')
    ax.set_ylabel('Data Asli')

    # Save confusion matrix plot to "image" folder
    image_folder = os.path.join(os.getcwd(), 'image')
    plt.savefig(os.path.join(image_folder, 'confusion_matrix.png'))

    st.pyplot(fig)

    # Tampilkan hasil menggunakan tabel DataFrame
    st.subheader("Data Hasil Klasifikasi")
    # Merge hasil prediksi dengan data asli berdasarkan indeks
    result_df = st.session_state['labeled_data'].iloc[indices_test].copy()
    result_df['Sentimen Data Prediksi'] = y_pred
    # Gabungkan token-token menjadi kalimat utuh
    result_df['Kata / Kalimat'] = result_df['final_prep'].apply(lambda x: ' '.join(x))
    # Pilih kolom yang akan ditampilkan
    result_df = result_df[['Kata / Kalimat', 'sentimen', 'Sentimen Data Prediksi']].rename(columns={'sentimen': 'Sentimen Data Asli'})
    st.dataframe(result_df, width=10000)

    # Hitung jumlah data yang berhasil di prediksi model & jumlah data yang salah di prediksi
    prediksi_benar = (y_pred == y_test).sum()
    prediksi_salah = (y_pred != y_test).sum()
    accuracy = prediksi_benar / (prediksi_benar + prediksi_salah) * 100

    st.write('Jumlah prediksi benar\t:', prediksi_benar)
    st.write('Jumlah prediksi salah\t:', prediksi_salah)
    st.write('Akurasi pengujian\t:', accuracy, '%')

    # Button to save classification result
    if st.button("Simpan Hasil Klasifikasi"):
        result_file_path = os.path.join("hasil", "result_classification.csv")
        result_df.to_csv(result_file_path, index=False)
        st.success(f"Hasil klasifikasi telah disimpan: {result_file_path}")

else:
    st.error("TF-IDF features not found. Please complete the TF-IDF extraction process first.")