import os
import sys
import joblib
import librosa
import numpy as np
import soundfile as sf

def load_model(model_path):
    return joblib.load(model_path)

def extract_features(file_path):
    y, sr = sf.read(file_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

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
