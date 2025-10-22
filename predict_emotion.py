#!/usr/bin/env python3
"""
Emotion Classifier - Web Compatible Version
Can be used in CLI or Streamlit Web App
"""

import sys
import numpy as np
import cv2
from tensorflow import keras
import os

# ---------------------- Load model ----------------------
MODEL_PATH = 'best_emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'emotion_classifier_final.h5'

print(f"Loading model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!\n")

emotion_names = ['Happy', 'Sad', 'Angry']
emotion_emojis = ['😊', '😢', '😠']

# ---------------------- Predict function ----------------------
def predict_emotion(image_input):
    """Predict emotion from image path OR NumPy image array"""

    # Handle both file path or NumPy array
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input

    if img is None:
        print("❌ Error: Cannot read image")
        return "Error"

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

    if len(faces) == 0:
        print("⚠️  No face detected, analyzing full image...")
        face = cv2.resize(img_gray, (48, 48))
    else:
        # Use largest face
        if len(faces) > 1:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        face = img_gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        print(f"✅ Face detected at ({x}, {y}) size: {w}x{h}")

    # Preprocess
    face_normalized = face.astype('float32') / 255.0
    face_input = face_normalized.reshape(1, 48, 48, 1)

    # Predict
    prediction = model.predict(face_input, verbose=0)
    winner_idx = np.argmax(prediction)
    confidence = prediction[0][winner_idx] * 100
    emotion = emotion_names[winner_idx]
    emoji = emotion_emojis[winner_idx]

    print(f"🎯 Predicted: {emotion} {emoji} ({confidence:.1f}% confidence)")
    return f"{emotion} {emoji} ({confidence:.1f}%)"

# ---------------------- CLI Mode ----------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_emotion.py <image_path>")
        print("Example: python predict_emotion.py test_face.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = predict_emotion(image_path)
    print(result)

