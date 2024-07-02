import streamlit as st

st.set_page_config(
    page_title="Analisis - About",
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

# Judul Aplikasi
st.title("Tentang Aplikasi")

# Keterangan Aplikasi
st.write("""
    Aplikasi ini bertujuan untuk menganalisis sentimen dari pengguna Twitter terhadap dengan kemenangan Timnas Indonesia pada Piala AsiaAFC 2024.
    """)

st.write("**Batasan Masalah Penelitian :**")
st.write("""
    1.	Data yang digunakan bersumber dari Twitter. Periode pengambilan data dari 1-30 April 2024. Data tersebut diambil dari Twitter dengan menggunakan API tweet-harvest yang bersumber dari https://helmisatria.com/blog/.
    2.	Menggunakan hastag **#TimnasDay** dan **#AFCU23AsianCup**.
    3.	Analisis sentimen terbatas pada kemenangan Tim Nasional Indonesia di pertandingan perempat final Piala Asia U-23 AFC 2024.
    4.	Sistem yang dibahas: \n
        a. Preprocessing dataset\n
        b. Klasifikasi ***Naive Bayes*** & Evaluasi ***Confusion Matrix*** \n
        c. Visualisasi sentimen\n
    5.	Klasifikasi sentimen bernilai positif dan negatif.
    """)

# Judul untuk Daftar Referensi
st.write("**Daftar Referensi :**")
# Daftar Referensi
st.write("""
    1. Kamus Inset (Indonesia Sentiment Lexicon), https://github.com/fajri91/InSet
    2. Kamus Normalisasi, https://github.com/ksnugroho/klasifikasi-spam-sms
    """)

# Pengembang Program
st.write("**Kontak :**")
st.write("""
    1. Nama        : Anwar Muzaki \n
    2. Institusi   : Universitas Nusantara PGRI Kediri, Fakultas Teknik dan Ilmu Komputer, Teknik Informatika \n
    3. Email       : zackcorporation2020@gmail.com \n
    """)