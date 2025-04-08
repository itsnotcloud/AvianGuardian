import os
import joblib
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, render_template
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

MODEL_PATH = r"C:\Users\Varun\OneDrive\Documents\Github\AvianGuardian\trained_model.pkl"
model = joblib.load(MODEL_PATH)

if not hasattr(model, "predict_proba"):
    raise ValueError("Model does not support confidence scores (predict_proba not available)")

# Twilio setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DESTINATION_PHONE_NUMBER = os.getenv("DESTINATION_PHONE_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

def send_notification(message):
    if not twilio_client:
        print("Twilio not configured - skipping notification")
        return
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=DESTINATION_PHONE_NUMBER
        )
        print("Notification sent successfully")
    except Exception as e:
        print(f"Error sending notification: {str(e)}")

def extract_features(file_path):
    try:
        y, sr = sf.read(file_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
        return np.mean(mfcc, axis=1).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Audio processing failed: {str(e)}")

def predict_audio(audio_file_path, latitude=None, longitude=None):
    try:
        features = extract_features(audio_file_path)
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        confidence = round(np.max(probabilities) * 100, 1)

        result = "Endangered" if prediction[0] == 1 else "Non-Endangered"

        if result == "Endangered":
            location = ""
            if latitude and longitude:
                location = f" at coordinates {latitude}, {longitude}"
            send_notification(f"ALERT: Endangered bird detected{location} (Confidence: {confidence}%)!")
        
        return result, confidence, latitude, longitude
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("error.html", error_message="No file provided"), 400

        file = request.files['file']
        if file.filename == '':
            return render_template("error.html", error_message="No file selected"), 400

        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')

        upload_dir = os.path.join(os.getcwd(), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        temp_path = os.path.join(upload_dir, file.filename)

        try:
            file.save(temp_path)
            result, confidence, lat, lng = predict_audio(temp_path, latitude, longitude)
            return render_template("result.html",
                                   result=result,
                                   confidence=confidence,
                                   latitude=lat,
                                   longitude=lng)
        except Exception as e:
            return render_template("error.html", error_message=str(e)), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
