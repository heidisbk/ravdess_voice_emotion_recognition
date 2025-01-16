import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
