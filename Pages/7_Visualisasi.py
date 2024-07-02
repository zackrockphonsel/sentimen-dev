import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from wordcloud import WordCloud
import os
from PIL import Image

st.set_page_config(
    page_title="Analisis - Visualisasi Sentimen",
    page_icon="📊",
    layout="wide"
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

st.title("Visualisasi Sentimen")

# Function to save plot to image folder
def save_plot(figure, filename):
    filepath = os.path.join("image", filename)
    figure.savefig(filepath)

# Check if labeled data is available in session state
# st.subheader('Visualisasi Sentimen Data Asli')
if 'labeled_data' in st.session_state:
    labeled_data = st.session_state['labeled_data']
    
    # # Plot sentiment distribution using a pie chart
    # sentiment_counts = labeled_data['sentimen'].value_counts()
    # fig1, ax1 = plt.subplots(1, 2, figsize=(12, 6))

    # # Plot pie chart for labeled data
    # ax1[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
    # ax1[0].set_title('Distribusi Sentimen Data Asli (Pie)')
    # ax1[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # # Plot bar chart for labeled data
    # ax1[1].bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
    # ax1[1].set_title('Distribusi Sentimen Data Asli (Bar Chart)')
    # ax1[1].set_xlabel('Sentimen')
    # ax1[1].set_ylabel('Count')
    # ax1[1].tick_params(axis='x', rotation=45)
    # ax1[1].grid(axis='y', linestyle='--', alpha=0.7)

    # # Save pie chart and bar chart to "image" folder
    # save_plot(fig1, 'sentimen_data_asli.png')

    # plt.tight_layout()
    # st.pyplot(fig1)

    # Load the Naive Bayes model from file
    naive_bayes_model = joblib.load('model/naive_bayes_model.pkl')

    # Make predictions using the loaded model
    st.subheader('Visualisasi Sentimen Data Prediksi')
    if 'tfidf_features' in st.session_state:
        X_test = st.session_state['tfidf_features']
        y_pred = naive_bayes_model.predict(X_test)

        # Plot sentiment distribution from prediction using a pie chart
        prediction_counts = pd.Series(y_pred).value_counts()
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))

        # Plot pie chart for prediction
        ax2[0].pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%')
        ax2[0].set_title('Distribusi Sentimen Prediksi (Pie)')
        ax2[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Plot Bar Chart for prediction
        ax2[1].bar(prediction_counts.index, prediction_counts.values, color='skyblue')
        ax2[1].set_title('Distribusi Sentimen Prediksi (Bar Chart)')
        ax2[1].set_xlabel('Sentimen')
        ax2[1].set_ylabel('Count')
        ax2[1].tick_params(axis='x', rotation=45)
        ax2[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Save pie chart and bar chart to "image" folder
        save_plot(fig2, 'sentimen_prediksi.png')

        plt.tight_layout()
        st.pyplot(fig2)

    else:
        st.error("Fitur TF-IDF tidak ditemukan. Silakan lakukan ekstraksi fitur terlebih dahulu di halaman Ekstraksi Fitur.")

    # Load the cloud image
    cloud_mask = np.array(Image.open('image/cloud.jpg'))

    # Generate WordCloud for positive sentiment
    st.subheader("WordCloud Sentimen Positif")
    positive_text = ' '.join(labeled_data[labeled_data['sentimen'] == 'positif']['final_prep'].explode().dropna().astype(str))
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white', mask=cloud_mask).generate(positive_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.title('WordCloud Sentimen Positif', size=10)
    plt.axis('off')

    # Save WordCloud to "image" folder
    save_plot(plt, 'wordcloud_sentimen_positif.png')

    st.pyplot(plt)

    # Generate WordCloud for negative sentiment
    st.subheader("WordCloud Sentimen Negatif")
    negative_text = ' '.join(labeled_data[labeled_data['sentimen'] == 'negatif']['final_prep'].explode().dropna().astype(str))
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white', mask=cloud_mask).generate(negative_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.title('WordCloud Sentimen Negatif', size=10)
    plt.axis('off')

    # Save WordCloud to "image" folder
    save_plot(plt, 'wordcloud_sentimen_negatif.png')

    st.pyplot(plt)

else:
    st.error("Data yang diberi label tidak ditemukan. Harap lakukan pelabelan terlebih dahulu di halaman Labelling.")
