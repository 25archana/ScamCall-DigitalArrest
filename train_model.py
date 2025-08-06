import os
import numpy as np
import librosa
import whisper
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Configuration
DATASET_DIR = "dataset"  # Must contain 'scam' and 'non_scam' subfolders
MODEL_PATH = "hybrid_xgb_model.json"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Initialize models
whisper_model = whisper.load_model("base")
audio_features, text_data, labels = [], [], []

print("Extracting audio and transcript features...")

for folder, label in [('scam', 1), ('non_scam', 0)]:
    path = os.path.join(DATASET_DIR, folder)
    for filename in tqdm(os.listdir(path), desc=f"{folder} files"):
        if not filename.lower().endswith('.wav'):
            continue

        file_path = os.path.join(path, filename)

        # Extract MFCC features
        try:
            y, sr = librosa.load(file_path, sr=None)
            if y.size == 0:
                continue
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0)
        except Exception as e:
            print(f"Audio feature error [{filename}]: {e}")
            continue

        # Transcribe using Whisper
        try:
            result = whisper_model.transcribe(file_path)
            transcript = result.get('text', '').strip()
            if not transcript:
                continue
        except Exception as e:
            print(f"Transcription error [{filename}]: {e}")
            continue

        audio_features.append(mfcc_mean)
        text_data.append(transcript)
        labels.append(label)

# TF-IDF Vectorization
print("Vectorizing transcripts...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
X_text = vectorizer.fit_transform(text_data).toarray()
X_audio = np.array(audio_features)

# Combine features and prepare labels
X = np.hstack((X_audio, X_text))
y = np.array(labels)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train XGBoost model
print("Training hybrid XGBoost model...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
model.save_model(MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"✅ Model saved: {MODEL_PATH}")
print(f"✅ Vectorizer saved: {VECTORIZER_PATH}")
