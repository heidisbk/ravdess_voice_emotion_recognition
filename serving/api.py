from fastapi import FastAPI, File, UploadFile, HTTPException
import random

# Initialisation de l'application FastAPI
app = FastAPI()

# Définir les émotions prédictibles
EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

# Endpoint pour vérifier l'état de l'API
@app.get("/")
def health_check():
    return {"status": "API is running"}

# Endpoint pour la prédiction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Vérification du type de fichier
        if file.content_type not in ["audio/wav", "audio/mpeg"]:
            raise HTTPException(status_code=400, detail="Fichier audio invalide. Seuls les formats WAV ou MP3 sont acceptés.")

        # Lecture du fichier audio (non utilisé ici)
        _ = await file.read()

        # Génération d'une prédiction aléatoire
        predicted_emotion = random.choice(EMOTIONS)
        confidence = random.uniform(0.5, 1.0)  # Confiance aléatoire entre 50% et 100%

        return {
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {e}")

# Exécution de l'application si appelée directement
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
