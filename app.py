import os
import joblib
import librosa
import numpy as np
import soundfile as sf
from flask import Flask, request, render_template_string
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

MODEL_PATH = r"C:\Users\Varun\OneDrive\Desktop\Bird Sound and Endangered Detection\trained_model.pkl"
model = joblib.load(MODEL_PATH)

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DESTINATION_PHONE_NUMBER = os.getenv("DESTINATION_PHONE_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_notification(message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=DESTINATION_PHONE_NUMBER
        )
        print("Notification sent!")
    except Exception as e:
        print(f"Error sending notification: {e}")

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

def predict_audio(audio_file_path):
    features = extract_features(audio_file_path)
    prediction = model.predict(features)
    result = "Endangered" if prediction[0] == 1 else "Non-Endangered"
    # If endangered, send an SMS alert
    if result == "Endangered":
        send_notification("Alert: Endangered bird detected!")
    return result

upload_page = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Bird Sound Endangered Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: #f8f9fa; }
      .container { max-width: 600px; margin-top: 50px; }
      .card { border: none; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
      .btn-custom { background-color: #007bff; color: #fff; }
      .btn-custom:hover { background-color: #0056b3; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card p-4">
        <h1 class="card-title text-center mb-4">Bird Sound Endangered Detection</h1>
        <form action="/" method="post" enctype="multipart/form-data">
          <div class="mb-3">
            <input type="file" name="file" class="form-control" accept="audio/*" required>
          </div>
          <div class="d-grid">
            <input type="submit" value="Upload and Predict" class="btn btn-custom">
          </div>
        </form>
      </div>
      <div class="text-center mt-3">
        &copy; 2025 Bird Sound Detection System
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

result_page = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Prediction Result</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body { background: #f8f9fa; }
      .container { max-width: 600px; margin-top: 50px; }
      .card { border: none; box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center; }
      h1 { font-size: 24px; color: #333; }
      a { text-decoration: none; color: #007bff; }
      a:hover { color: #0056b3; }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card p-4">
        <h1>Prediction: {{ result }}</h1>
        <p class="mt-3"><a href="/">Upload Another File</a></p>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file provided", 400
        file = request.files['file']
        if file.filename == "":
            return "No file selected", 400
        
        temp_file_path = "temp_audio.wav"
        file.save(temp_file_path)
        try:
            result = predict_audio(temp_file_path)
        except Exception as e:
            result = f"Error during prediction: {str(e)}"
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        return render_template_string(result_page, result=result)
    return render_template_string(upload_page)

if __name__ == "__main__":
    app.run(debug=True)
