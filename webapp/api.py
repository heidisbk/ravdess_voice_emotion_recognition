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
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        # local
        # response = requests.post("http://localhost:8080/predict", files=files)

        # docker
        response = requests.post("http://serving-api:8080/predict", files=files)

        if response.status_code == 200:
            st.write("Prédiction :", response.json().get("emotion"))
            st.write("Confiance :", response.json().get("confidence"))
            st.write("Probabilités :", response.json().get("probabilities"))
        else:
            st.write("Erreur :", response.text)

        # traitement du fichier audio
        st.write("Traitement du fichier audio")
