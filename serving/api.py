from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import random
import numpy as np
from tempfile import NamedTemporaryFile
import joblib
import pickle
import os
import csv
import time

import librosa

import tensorflow as tf
import tensorflow.keras as keras

# Initialisation de l'application FastAPI
app = FastAPI()


# Charger le modèle pré-entrainé
MODEL_PATH = "../artifacts/model.pkl"
SCALER_PATH = "../artifacts/scaler.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle ou du scaler : {e}")

# Définir les émotions prédictibles
EMOTION_MAP = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

def extract_features_advanced(file_path):
    # Charger le fichier audio
    audio, sr = librosa.load(file_path, sr=None)  
    
    # MFCCs : 100 coefficients
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=100)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Chroma STFT : Capturer l'énergie distribuée à travers les 12 tons
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_stft_scaled = np.mean(chroma_stft.T, axis=0)
    
    # Spectrogramme de puissance (Mel Spectrogram) : 128 valeurs
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram)
    mel_spectrogram_scaled = np.mean(mel_spectrogram_db.T, axis=0)
    
    # STFT (Short-Time Fourier Transform) : 1025 valeurs
    stft = np.abs(librosa.stft(audio))
    stft_scaled = np.mean(stft.T, axis=0)
    
    # Concaténer toutes les caractéristiques
    combined_features = np.hstack([mfccs_scaled, chroma_stft_scaled, mel_spectrogram_scaled, stft_scaled])
    
    return combined_features

def preprocess_and_predict(file_path: str) -> dict:
    """
    1) Extrait les features du fichier audio `file_path`.
    2) Transforme avec le scaler.
    3) Fait la prédiction avec le modèle Keras.
    4) Retourne le label prédit, la confiance, etc.
    """
    # 1) Extraire les features
    features = extract_features_advanced(file_path)  # shape (1265,)
    
    # 2) Mise en forme
    features_reshaped = features.reshape(1, -1)  # (1, 1265)
    
    # 3) Scale
    features_scaled = scaler.transform(features_reshaped)  # (1, 1265)
    
    # 4) Prédiction
    probabilities = model.predict(features_scaled)[0]  # shape (8,)
    predicted_class = np.argmax(probabilities)
    predicted_emotion = EMOTION_MAP[predicted_class]
    confidence = float(probabilities[predicted_class])  # un float pour la classe prédite
    
    # Convertir en liste "classique" pour JSON
    probabilities_list = probabilities.tolist()
    features_list = features.tolist()

    # 5) Sauvegarde dans le CSV
    # Ici, tu souhaites sauvegarder toutes les features + label prédit + proba associée
    # => On ajoute dans "prod_data.csv" :
    #    [f0, f1, f2, ..., fN, label_predit, proba_predite]
    # Pour stocker TOUTES les probabilités, tu peux décider de les ajouter
    # ou te limiter à la plus haute. Fais comme tu préfères.
    
    row_to_save = list(features_list)  # toutes les features
    row_to_save.append(predicted_emotion)
    row_to_save.append(confidence)
    # Ou ajouter toutes les probabilités
    # row_to_save.extend(probabilities_list)

    # Ouvrir/fermer le CSV en mode 'a' (append)
    csv_path = "/data/prod_data.csv"  # ton chemin dans le conteneur (monté en volume)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Optionnel : écriture d’un header si le fichier n’existe pas encore
        if not file_exists:
            # Construire un header rudimentaire
            header = [f"feature_{i}" for i in range(len(features_list))]
            header.append("prediction")   # predicted_emotion
            header.append("confidence")
            writer.writerow(header)
        writer.writerow(row_to_save)

    # 6) Retour
    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities_list,
    }

# Endpoint pour vérifier l'état de l'API
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Endpoint pour la prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(
                status_code=400, 
                detail="Fichier audio invalide. Seuls les formats WAV ou MP3 sont acceptés."
            )
        
        # Nom de fichier temporaire, par ex. ajout d'un timestamp
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
        
        # Appel de la fonction de prédiction
        results = preprocess_and_predict(temp_file_path)

        # Nettoyage du fichier temporaire
        os.remove(temp_file_path)

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")

@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    target: str = Form(...),
    prediction: str = Form(...)
):
    """
    1) Ré-extrait les features de l'audio.
    2) Ajoute au CSV une ligne contenant:
       - toutes les features
       - la prédiction
       - la vraie étiquette (feedback)
    """
    try:
        # Vérifier le type
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(
                status_code=400,
                detail="Fichier audio invalide. Seuls WAV ou MP3."
            )

        # Créer un fichier temporaire
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        # Extraire les features
        features = extract_features_advanced(temp_file_path)
        os.remove(temp_file_path)
        features_list = features.tolist()

        # Ajouter au CSV
        file_exists = os.path.isfile(CSV_PATH)
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Si le fichier n’existe pas encore, on écrit l’en-tête
            if not file_exists:
                nb_features = len(features_list)
                header = [f"feature_{i}" for i in range(nb_features)]
                header += ["prediction", "target"]
                writer.writerow(header)

            # Crée la ligne avec features + prediction + target
            row_to_save = features_list + [prediction, target]
            writer.writerow(row_to_save)

        return {"status": "feedback enregistré avec succès"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exécution de l'application si appelée directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
