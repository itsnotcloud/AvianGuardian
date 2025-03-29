import os
import joblib
import librosa
import numpy as np
import soundfile as sf
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Define Prediction Functions ---

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

class BirdSoundApp:
    def __init__(self, master):
        self.master = master
        master.title("Bird Sound Endangered Detection")
        
        self.model_path = r"C:\Users\Varun\OneDrive\Desktop\Bird Sound and Endangered Detection\trained_model.pkl"
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.master.destroy()

        self.label = tk.Label(master, text="Select an audio file to predict:")
        self.label.pack(pady=10)

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.pack(pady=5)

        self.predict_button = tk.Button(master, text="Predict", command=self.make_prediction, state=tk.DISABLED)
        self.predict_button.pack(pady=5)

        self.result_label = tk.Label(master, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        self.audio_file = None

    def browse_file(self):
        filetypes = [("WAV files", "*.wav"), ("MP3 files", "*.mp3"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select an Audio File", filetypes=filetypes)
        if filename:
            self.audio_file = filename
            self.label.config(text=f"Selected file:\n{os.path.basename(filename)}")
            self.predict_button.config(state=tk.NORMAL)

    def make_prediction(self):
        if self.audio_file is None:
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        try:
            result = predict_audio(self.model, self.audio_file)
            self.result_label.config(text=f"Prediction: {result}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BirdSoundApp(root)
    root.mainloop()
