import streamlit as st
import pandas as pd
import time

st.set_page_config(
    page_title="Analisis - Upload Dataset",
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

st.title("Upload Dataset")

st.subheader('Load Dataset Mentah')

# Tambahkan komponen untuk mengunggah file dataset
uploaded_file = st.file_uploader("Unggah file dataset (CSV atau Excel)", type=["csv", "xlsx"])

# Initialize session_state if not already done
if 'data' not in st.session_state:
    st.session_state['data'] = None
    st.session_state['execution_time'] = None

if uploaded_file is not None:
    try:
        start_time = time.time()
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)

        end_time = time.time()
        execution_time = end_time - start_time
        st.success(f'Done! Waktu Eksekusi: {execution_time:.2f} detik')
        st.session_state['data'] = data
        st.session_state['execution_time'] = execution_time
        st.write("Dataset berhasil diunggah!")
        st.dataframe(data.head(10), width=10000)

        # Informasi tentang dataset
        st.write("Informasi Dataset:")
        st.write(f"Jumlah Baris: {data.shape[0]}")
        st.write(f"Jumlah Kolom: {data.shape[1]}")
        st.write("Tipe Dataset: DataFrame")
        
    except Exception as e:
        st.write("Terjadi kesalahan saat mengunggah file: ", e)
            
    # Menampilkan opsi untuk menghapus kolom jika dataset telah diunggah
    if st.session_state['data'] is not None:
        st.subheader('Hapus Kolom yang Tidak Diperlukan')

        # Pilih kolom yang ingin dihapus
        columns_to_drop = st.multiselect('Pilih kolom yang ingin dihapus:', st.session_state['data'].columns)

        # Tombol untuk menghapus kolom
        if st.button('Hapus Kolom'):
            st.session_state['data'].drop(columns=columns_to_drop, inplace=True)
            st.write("Kolom yang dipilih telah dihapus!")
            st.dataframe(st.session_state['data'].head(10), width=10000)
            
            # Informasi tentang dataset setelah penghapusan kolom
            st.write("Informasi Dataset Setelah Penghapusan Kolom:")
            st.write(f"Jumlah Baris: {st.session_state['data'].shape[0]}")
            st.write(f"Jumlah Kolom: {st.session_state['data'].shape[1]}")
            st.write("Tipe Dataset: DataFrame")