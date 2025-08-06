#  Digital Arrest Scam Detector

A hybrid machine learning web application for detecting scam calls using audio and text analysis, specifically targeting "digital arrest" scam scenarios.

##  Features

- Upload `.wav` audio files to analyze scam likelihood.
- Detects key "digital arrest" scam keywords from transcribed audio.
- Visualizes recent scam locations via interactive map (Leaflet.js).
- Embedded awareness videos and scam news carousel.
- Real-time scam alerts and statistics.

##  Project Structure

├── app.py # Flask web app
├── train_model.py # Script to train hybrid model (audio + text)
├── hybrid_xgb_model.json # Saved XGBoost model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── dataset/ # Audio dataset (subfolders: scam, non_scam)
├── static/ # Static files (images, css, js, uploads)
│ ├── css/
│ ├── js/
│ ├── uploads/ # Uploaded audio files
│ ├── *.png/jpg/jpeg # Images for site
├── templates/ # HTML templates (index.html, upload.html)
└── README.md # This file


## Model Overview

- **Model**: Hybrid XGBoost Classifier
- **Features**:
  - MFCC (audio)
  - TF-IDF (transcript text from Whisper ASR)
- **Trained on**: Custom dataset with labeled scam/non-scam `.wav` files.


**Tech Stack**
Frontend: Bootstrap, Leaflet.js, AOS

Backend: Flask, Whisper ASR

ML: XGBoost, Librosa, TF-IDF

