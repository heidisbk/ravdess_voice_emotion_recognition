import streamlit as st
import requests

# Interface utilisateur
st.title("Interface de Prédiction des Émotions")
st.write("Uploadez un fichier audio pour prédire l'émotion.")

uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("Prédire l'émotion"):
        # Envoi du fichier à l'API de serving
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://serving-api:8080/predict", files=files)

        if response.status_code == 200:
            st.write("Prédiction :", response.json().get("emotion"))
        else:
            st.write("Erreur :", response.text)
