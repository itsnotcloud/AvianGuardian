import os
import sys
import joblib
import librosa
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def extract_features(file_path):

    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        features = np.concatenate([
            np.mean(mfcc, axis=1),         
            [np.mean(spectral_centroid)],
            [np.mean(zero_crossing_rate)],
            np.mean(chroma, axis=1)
        ])

        return features.reshape(1, -1)

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def predict_audio(model, audio_file):
    features = extract_features(audio_file)
    prediction = model.predict(features)
    return "Endangered" if prediction[0] == 1 else "Non-Endangered"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        audio_file = r"C:\Users\Varun\OneDrive\Desktop\Bird Sound and Endangered Detection\processed_data\Common Wood Pigeon_test_XC536577 1.wav"
        print("No audio file provided. Using default:", audio_file)
    else:
        audio_file = sys.argv[1]
    
    model_path = r"C:\Users\Varun\OneDrive\Desktop\Bird Sound and Endangered Detection\trained_model.pkl"
    model = load_model(model_path)
    result = predict_audio(model, audio_file)
    print("Prediction:", result)