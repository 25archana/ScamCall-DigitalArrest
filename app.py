import os
import re
import numpy as np
import librosa
import whisper
import joblib
from flask import Flask, render_template, request
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
model = XGBClassifier()
model.load_model("hybrid_xgb_model.json")  # Hybrid model with audio + text features

vectorizer = joblib.load("tfidf_vectorizer.pkl")  # TF-IDF for text features
whisper_model = whisper.load_model("base")        # Whisper model for transcription

# Keywords for scam detection
digital_arrest_keywords = [
    'arrest', 'warrant', 'police', 'court', 'fine', 'digital arrest',
    'legal', 'fraud', 'investigation', 'narcotics', 'money laundering', 'cyber crime',
    'identity', 'trafficking', 'customs', 'border', 'passport', 'visa', 'immigration',
    'fake documents', 'interpol', 'drug', 'parcel', 'international', 'blog trafficking',
    'national security', 'suspicious activity', 'social security', 'irs', 'password', 'otp'
]

# Extract MFCC audio features
def extract_audio_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            raise ValueError("Empty audio file.")
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Audio feature extraction error: {e}")
        return None

# Preprocess text for analysis
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html', error="No file selected.")

        if not file.filename.lower().endswith('.wav'):
            return render_template('upload.html', error="Invalid file format. Upload a WAV file.")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        audio_features = extract_audio_features(filepath)
        if audio_features is None:
            return render_template('upload.html', error="Failed to extract audio features.")

        result = whisper_model.transcribe(filepath)
        transcript = result.get('text', '').strip()

        if not transcript:
            return render_template('upload.html', error="Transcription failed or audio was empty.")

        clean_text = preprocess_text(transcript)
        text_features = vectorizer.transform([clean_text]).toarray()[0]

        combined_features = np.concatenate((audio_features, text_features)).reshape(1, -1)
        prediction = model.predict(combined_features)[0]
        label = 'Scam' if prediction == 1 else 'Legitimate'

        detected_keywords = [
            kw for kw in digital_arrest_keywords
            if re.search(r'\b' + re.escape(kw) + r'\b', transcript.lower())
        ]

        is_digital_arrest = bool(detected_keywords)
        if is_digital_arrest:
            label = 'Scam'

        result_msg = "Scam Call Detected!"
        if label == 'Legitimate':
            result_msg = "Legitimate Call."
        elif is_digital_arrest:
            result_msg += " Digital Arrest Call Detected."

        return render_template(
            'upload.html',
            transcript=transcript,
            result=result_msg,
            keywords=detected_keywords
        )

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
