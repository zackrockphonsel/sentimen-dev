import streamlit as st
import pandas as pd
import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from datetime import datetime
from sqlalchemy import create_engine

st.set_page_config(
    page_title="Analisis - Preprocessing",
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

# Fungsi untuk memuat kamus normalisasi dari database MySQL menggunakan SQLAlchemy
def load_normalization():
    try:
        # Koneksi ke database MySQL menggunakan SQLAlchemy
        engine = create_engine('mysql+pymysql://root:@localhost/db_skripsi')
        
        # Eksekusi query dan memuat data ke DataFrame
        query = "SELECT singkat, hasil FROM key_norm"
        df = pd.read_sql(query, engine)
        
        # Pastikan DataFrame tidak kosong
        if df.empty:
            st.error("Data normalisasi tidak ditemukan di database.")
            return None
        
        # Konversi DataFrame ke dictionary
        normalization_dict = pd.Series(df.hasil.values, index=df.singkat).to_dict()

        # Tambahkan korpus normalisasi manual
        manual_normalization = {
            'bnyk': 'banyak', 'big': 'besar', 'joint': 'gabung', 'abngkuw': 'abangku','sejarha':'sejarah','akunne':'akun','and':'dan','atlitttttt':'atlet',
            'brp': 'berapa', 'congrast': 'selamat','vs': 'versus','skuad': 'squad', 'el klemer':'el klemer','elkan baggot':'elkan baggot',
            'play off':'pertandingan ulang', 'kick off':'memulai', 'playoff':'pertandingan ulang','kickoff':'memulai','apresiasiin':'apresiasi',
            'aamiinn':'amiin','abangkuuu':'abangku','adl':'adalah','adlh':'adalah','ado':'ada','ajg':'anjing','amazingggg':'amazing','bdmd':'badmood',
            'anj':'anjing','anjeeeeeng':'anjing','anjinggg':'anjing','anjirr':'anjir','anjj':'anjir','anjrit':'anjir','argentinaa':'argentina','arhannn':'arhan','atiati':'hati hati','aus':'haus','ayooo':'ayo','baek':'baik',
            'bahagianyaaaaa':'bahagia','bangeeet':'banget','bangeets':'banget','bangett':'banget','bangettt':'banget','bangetttt':'banget',
            'bangetttttttt':'banget','banggaaaa':'bangga','banggaaaaa':'bangga','banggaaaaaa':'bangga','banggaaaaaaaa':'bangga','banggan':'bangga','banggat':'bangga','bb':'berat badan','bejirrrrrr':'anjir','berbahagiaaaaaaaaaaaaaaaaaaaaa':'berbahagia',
            'berbahagialahh':'berbahagialah','berkahhh':'berkah','bgi':'bagi','bgini':'bagini','bgtt':'banget','bgttt':'banget','bgtttt':'banget',
            'bingits':'banget','bjirlah':'bjir','bjirr':'bjir','body':'badan','bojone':'istrinya','book':'buku','boss':'bos','bosss':'bos','bossss':'bos',
            'bossssss':'bos','broow':'bro','cakepp':'cakep','cominggg':'coming','congratssssss':'congrats','crying':'menangis','datanggggg':'datang',
            'dikei':'dikasih','dimanaa':'dimana','dimanamana':'dimana mana','dino':'hari','finaaaaallllllllllll':'final','finalti':'pinalti','finalty':'pinalti','finall':'final',
            'fjnalll':'final','gasssss':'gas','gilaaaa':'gila','gilaaaaa':'gila','gilaaaaaaa':'gila','gilaaaaak':'gila','gilasih':'gila sih','gilee':'gila','gilllaa':'gila',
            'gituu':'begitu','guee':'gue','indonesiaa':'indonesia','indonesiaaaa':'indonesia','indonesiaaaaaakulonuwun':'indonesia','indonesiasukses':'indonesia sukses',
            'indoo':'indonesia','indo':'indonesia','abisss':'habis','abooottttt':'berat','additional':'tambahan','alhamdhlillah':'alhamdulillah','alhamdulillaaah':'alhamdulillah','alhamdulillahhh':'alhamdulillah','alhamdulillahhhh':'alhamdulillah',
            'allahhh':'allah','allahhhh':'allah','allaholympide':'allah olympiade','allahu':'allah','alloh':'allah','anjiiiiiing':'anjing','anjiingggg':'anjing','apaa':'apa',
            'apaaa':'apa','apaapa':'apa','apanii':'apa ini','asiaaaa':'asia','asiaaaaa':'asia','asliii':'asli','aspekdi':'aspek di','assalamu':'assalamualaikum','bct':'bacot',
            'ayok':'ayo','ayolahhh':'ayo','ayoo':'ayo','ayoooooo':'ayo','babikkk':'babi','banggaa':'bangga','banggan':'bangga','banggaterima':'bangga terima','banggatetap':'bangga tetap',
            'bgsat':'bangsat','bgst':'bangsat','bgsttt':'bangsat','bnyak':'banyak','bossu':'bos','bruntung':'beruntung','buanggaa':'bangga','buangettt':'banget','bubaaaaarrrr':'bubar','bubaaarbubaaar':'bubar bubar',
            'claaaaaaaaaaaaaaaudia':'claudia','ya':'iya','yaa':'iya','yaaa':'iya','yaaaa':'iya','yaaaaa':'iya','abaanggkuuu':'abangku','abangkuh':'abangku','abangan':'abang',
            
            # Tambahkan kata-kata lain yang perlu dinormalisasi di sini
        }

        normalization_dict.update(manual_normalization)
        return normalization_dict
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Muat kamus normalisasi dari database
normalization_dict = load_normalization()

# Fungsi untuk normalisasi teks
def normalize_text(tokens, normalization_dict):
    # Jika kamus normalisasi tidak ada, kembalikan token asli
    if normalization_dict is None:
        return tokens
    # Normalisasi kata menggunakan kamus
    words = [normalization_dict.get(word, word) for word in tokens]
    return words

# Download stopwords from NLTK
nltk.download('stopwords')

# Load stopwords from NLTK
stopwords_nltk = set(stopwords.words('indonesian'))

# Stopwords manual
stopwords_manual = [
    'wkwkwkwk','wa','ya','gtgt','gacor','anjing','an','anjir','anw','anu','bah','bajing','bar','bara','be','bek','belyke','bta','aingg','aishh','aka','al','am',
    'kah','nya','amp','rt','k','follow','like','ar','are','asn','asu','at','atan','anjir','bjir','bubble','bulol','cak','cenasuuuu','alla','alr','almulk','altoxen',
    'ni','betslot','id','ya','stade','leo','as','poissy','de','awokawokawokawok','chuaaakksss','charles','chukkae','cie','cok','cokkk','anjiirrrr','anjim','anjink',
    'anjirnrbrgehehsgsggggggg','anjirrrr','anjjj','anthem','anyiinggg','anyingg','ao','ap','aokwaokwaokw','arghhhhhhhhgggg','argh','artistaasiatico','asbwi','aseppp',
    'asianabet','asiasat','asuuu','atuh','au','avc','awi','ayong','ayu','aza','bacoott','bacot','badjingan','bal','baleg','bangil','bangke','bat','batt','beng','ber',
    'coooook','mank','leh','aa','aaaaa','aaaaaaaa','aaaaaabetapa','adq','ador','aepep','aet','ahhhh','ahkk','ahjussi','add','ae','ah','ahay','ai','ahyeon','bg','bgs',
    'coy','coz','cung','cup','cut','cuy','dag','dagdigdugser','deg','degan','deh','dek','demgan','den','deng','dig','dih','dik','dislorokke','dlp','beib','beljek',
    'dm','dna','dob','doi','doha','done','donggg','dpc','dug','dukun','duarr','dugun','dul','eeee','eeeee','eh','ehhh','el','et','euy','ez','bb','bday','bha','bhut','bin',
    'friksi','ft','fuckin','fyp','gbd','ge','gefuk','ges','gess','gih','gin','gila','ginseng','gingseng','gisune','go','goblokgoblokin','gtgtgtgt','bio','bnnay','blumm','bnwt',
    'gtgtgtgtjasa','gtgtgtjasa','guajg','gwenchana','gws','gwww','haag','hahahaha','hahayyy','hahah','haiiiii','hail','idn','idzes','if','ig','ih','ikan','bo','bot','br','bu','brt','btc',
    'in','ina','inaa','ini','iniblm','iniswlamat','is','isi','isiin','it','itachi','ivar','je','jay','jingaaaaaaan','jin','jingkrak','jir','jirr','jisung','bsd','bts',
    'jt','ka','jusa','junat','jungkok','jumar','jum','jumawa','jungkook','kabirov','kahhh','kelaaaaassszz','aaa','aaliyah','aamiinwopyu','aaron','aboh','abroad','ac','acakadut',
    'buajingannnn','cc','ceu','cf','cfa','chu','choi','choe','cik','cileunyi','ckd','clan','cn','cmf','cm','coi','cp','croot','cs','cv','cvt','wkkwk','wkkwkwkw','wkw','wkwk','wkwkkw',
    'wkwkw','wkwkwk','wkwkwkkwkw','wkwkwkw','wkwkwkwkw','wkwkwkwkwk','wkwowkwowkwowk','wo','woe','woee','woeee','ws','wss','wwf','xavi','xda','xi','xiomi','xl','xt','xxx','yaaahh','yaahh',
    'yaxshi','ye','yeaaahh','yeaaay','yeaayyyyyyyyyy','yeahh','yeay','yee','yeeah','yelah','yen','ygy','yh','yha','yin','yll','ytta','yu','yuh','yuhhuuuu','yuhuuu','yuk','yukk','yukyukkk','yuu',
    'yuuukkkk','yuuu','yuukk','slot'
    # Tambahkan kata-kata lain yang ingin di-stopword di sini
]

# Gabungkan stopwords manual dengan stopwords dari NLTK
stopwords_total = stopwords_nltk.union(stopwords_manual)

# Stemming dengan Sastrawi
def stem_text(tokens):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Stem setiap token
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(stemmed_tokens)

st.title("Preprocessing")

# Memeriksa apakah data telah diproses sebelumnya
if 'preprocessed_data' not in st.session_state:
    # Proses preprocessing
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        st.write("Di sini Anda dapat melakukan preprocessing data.")

        # Cleaning
        st.subheader("1. Cleaning")
        start_time_cleaning = time.time()
        data['cleaned'] = data.iloc[:, 0].apply(lambda x: re.sub(r'http\S+|www.\S+|\#\w+\s*', '', str(x)))  # Ensure x is converted to string
        data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'@\w+', '', str(x)))  # Remove mentions
        data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))  # Remove punctuation
        data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'\d+', '', str(x)))  # Remove numbers
        end_time_cleaning = time.time()
        execution_time_cleaning = end_time_cleaning - start_time_cleaning
        st.success(f'Done! Waktu Eksekusi: {execution_time_cleaning:.2f} detik')

        st.dataframe(data[['cleaned']].head(), width=10000)

        # Casefolding
        st.subheader("2. Casefolding")
        start_time_casefolding = time.time()
        data['casefolded'] = data['cleaned'].apply(lambda x: str(x).lower())  # Ensure x is converted to string
        end_time_casefolding = time.time()
        execution_time_casefolding = end_time_casefolding - start_time_casefolding
        st.success(f'Done! Waktu Eksekusi: {execution_time_casefolding:.2f} detik')

        st.dataframe(data[['casefolded']].head(), width=10000)

        # Tokenizing
        st.subheader("3. Tokenizing")
        start_time_tokenizing = time.time()
        data['tokenized'] = data['casefolded'].apply(word_tokenize)
        end_time_tokenizing = time.time()
        execution_time_tokenizing = end_time_tokenizing - start_time_tokenizing
        st.success(f'Done! Waktu Eksekusi: {execution_time_tokenizing:.2f} detik')

        st.dataframe(data[['tokenized']].head(), width=10000)

        # Normalisasi menggunakan kamus normalisasi dari database
        st.subheader("4. Normalisasi")
        start_time_normalization = time.time()
        data['normalized'] = data['tokenized'].apply(lambda tokens: normalize_text(tokens, normalization_dict))
        end_time_normalization = time.time()
        execution_time_normalization = end_time_normalization - start_time_normalization
        st.success(f'Done! Waktu Eksekusi: {execution_time_normalization:.2f} detik')

        st.dataframe(data[['normalized']].head(), width=10000)

        # Stopword Removal
        st.subheader("5. Stopword Removal")
        start_time_stopword = time.time()
        data['stopword_removed'] = data['normalized'].apply(lambda tokens: [word for word in tokens if word not in stopwords_total])
        end_time_stopword = time.time()
        execution_time_stopword = end_time_stopword - start_time_stopword
        st.success(f'Done! Waktu Eksekusi: {execution_time_stopword:.2f} detik')

        st.dataframe(data[['stopword_removed']].head(), width=10000)

        # Stemming
        st.subheader("6. Stemming (Sastrawi)")
        st.info('Proses Stemming membutuhkan waktu yang cukup lama, bergantung pada banyaknya data', icon="ℹ️")
        start_time_stemming = time.time()
        data['stemmed_text'] = data['stopword_removed'].apply(stem_text)
        end_time_stemming = time.time()
        execution_time_stemming = end_time_stemming - start_time_stemming
        st.success(f'Done! Waktu Eksekusi: {execution_time_stemming:.2f} detik')

        st.dataframe(data[['stemmed_text']].head(50), width=10000)

        # Final Preprocessing
        data['final_prep'] = data['stemmed_text'].apply(word_tokenize)
        # data['final_prep'] = data['stopword_removed'].apply(lambda tokens: [word for word in tokens if word not in stopwords_nltk])
        # st.dataframe(data[['final_prep']].head(50), width=10000)

        # Simpan data yang telah diproses ke dalam session state
        st.session_state['preprocessed_data'] = data

        # Button to save preprocessing result
        if st.button("Simpan Hasil Preprocessing"):
            result_file_path = os.path.join("hasil", "result_preprocessing.csv")
            data.to_csv(result_file_path, index=False)
            st.success(f"Hasil preprocessing telah disimpan: {result_file_path}")

    else:
        st.error("Harap unggah dataset terlebih dahulu melalui menu 'Upload Dataset'.")
else:
    # Jika data sudah diproses sebelumnya, tampilkan hasilnya
    data = st.session_state['preprocessed_data']
    # Show a progress bar
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.03)  # Simulate some computation
        progress_bar.progress(percent_complete + 1)

    st.write("Data sudah diproses sebelumnya.")
    st.write("1. Cleaning")
    if 'cleaned' in data:
        st.dataframe(data[['cleaned']].head(50), width=10000)
    else:
        st.write("Data cleaning belum tersedia.")
    st.write("2. Casefolding")
    if 'casefolded' in data:
        st.dataframe(data[['casefolded']].head(50), width=10000)
    else:
        st.write("Data casefolding belum tersedia.")
    st.write("3. Tokenize")
    if 'tokenized' in data:
        st.dataframe(data[['tokenized']].head(50), width=10000)
    else:
        st.write("Data tokenizing belum tersedia.")
    st.write("4. Normalisasi")
    if 'normalized' in data:
        st.dataframe(data[['normalized']].head(50), width=10000)
    else:
        st.write("Data normalisasi belum tersedia.")
    st.write("5. Stopword Remove")
    if 'stopword_removed' in data:
        st.dataframe(data[['stopword_removed']].head(50), width=10000)
    else:
        st.write("Data stopword removal belum tersedia.")
    st.write("6. Stemming")
    if 'stemmed_text' in data:
        st.dataframe(data[['stemmed_text']].head(50), width=10000)
    else:
        st.write("Data stemming belum tersedia.")
    st.write("7. Final Preprocessing")
    if 'final_prep' in data:
        st.dataframe(data[['final_prep']].head(50), width=10000)
    else:
        st.write("Data final preprocessing belum tersedia.")

# Button to save preprocessing result
    if st.button("Simpan Hasil Preprocessing"):
        result_file_path = os.path.join("hasil", "result_preprocessing.csv")
        data.to_csv(result_file_path, index=False)
        st.success(f"Hasil preprocessing telah disimpan: {result_file_path}")