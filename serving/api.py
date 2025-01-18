from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import random
import numpy as np
from tempfile import NamedTemporaryFile
import joblib
import pickle
import os
import csv
import time
from pydantic import BaseModel
from typing import List

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
                detail="Fichier audio invalide. Seuls WAV ou MP3 sont acceptés."
            )
        
        temp_file_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())
        
        features = extract_features_advanced(temp_file_path)
        os.remove(temp_file_path)

        # Mise en forme pour le modèle
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)

        # Prédiction
        probabilities = model.predict(features_scaled)[0]
        predicted_class = np.argmax(probabilities)
        predicted_emotion = EMOTION_MAP[predicted_class]
        confidence = float(probabilities[predicted_class])

        # Conversion pour JSON
        probabilities_list = probabilities.tolist()
        features_list = features.tolist()

        # On ne write plus rien dans prod_data.csv ici.
        return {
            "features": features_list,
            "prediction": predicted_emotion,
            "confidence": confidence,
            "probabilities": probabilities_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")


class FeedbackRequest(BaseModel):
    features: List[float]
    prediction: str
    target: str

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    """
    On reçoit:
     - features (Liste de float)
     - prediction (str)
     - target (str)
    Puis on écrit dans /data/prod_data.csv une ligne:
       feature_0, feature_1, ..., feature_n, prediction, target
    """
    try:
        features_list = req.features
        prediction = req.prediction
        target = req.target

        csv_path = "/data/prod_data.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = [f"feature_{i}" for i in range(len(features_list))]
                header += ["prediction", "target"]
                writer.writerow(header)

            row_to_save = list(features_list) + [prediction, target]
            writer.writerow(row_to_save)

        return {"status": "feedback enregistré avec succès"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exécution de l'application si appelée directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
