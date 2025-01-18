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
            pred_data = response.json()
            st.write("Prédiction :", pred_data.get("emotion"))
            st.write("Confiance :", pred_data.get("confidence"))
            st.write("Probabilités :", pred_data.get("probabilities"))
            # Stockage local de la prédiction
            st.session_state["last_prediction"] = pred_data
            st.session_state["last_audio"] = files  # pour pouvoir le réutiliser
        else:
            st.write("Erreur :", response.text)

        # traitement du fichier audio
        st.write("Traitement du fichier audio")

    # Champ texte + bouton pour le feedback
    true_label = st.text_input("Si vous connaissez la vraie émotion, entrez-la ici :")
    if st.button("Envoyer le feedback"):
        if "last_prediction" in st.session_state:
            # On réutilise la prédiction + le fichier audio
            pred_data = st.session_state["last_prediction"]
            audio_data = st.session_state["last_audio"]  # le fichier initial

            # Prépare la requête
            data = {
                "true_label": true_label,
                "predicted_label": pred_data["emotion"]
            }
            files_feedback = {"file": audio_data["file"]}

            feedback_response = requests.post(
                "http://localhost:8080/feedback",
                data=data,
                files=files_feedback
            )
            if feedback_response.status_code == 200:
                st.success("Feedback enregistré avec succès !")
            else:
                st.error(f"Erreur feedback: {feedback_response.text}")
        else:
            st.warning("Aucune prédiction disponible pour donner un feedback.")
