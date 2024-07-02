import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen Twitter",
    page_icon="🚀",
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

st.title("Analisis Sentimen Twitter")

st.subheader("Analisis Sentimen Twitter (X) Terhadap Kemenangan Timnas U-23")
st.write("""
    Aplikasi ini bertujuan untuk menganalisis sentimen dari pengguna Twitter terhadap dengan kemenangan Timnas Indoensia pada Piala AsiaAFC 2024. \n
    Data Tweet dicari berdasarkan tagar (hastags) **#TimnasDay** & **#AFCUAsianCup**, \n
    Algoritma yang digunakan pada aplikasi yaitu ***Naive Bayes***. \n
    """)
