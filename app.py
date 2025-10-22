import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import tempfile
import os

# --- Page setup ---
st.set_page_config(
    page_title="Aurora Emotion Classifier",
    page_icon="aurora_logo.png",
    layout="centered"
)

# --- Background CSS (Aurora gradient style) ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1a1a40, #1a1a2e, #16213e, #0f3460);
        background-attachment: fixed;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    }
    h1 {
        color: #00ffff;
        text-align: center;
        font-size: 2.5rem;
        letter-spacing: 2px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
MODEL_PATH = 'best_emotion_model.h5'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'emotion_classifier_final.h5'

st.sidebar.success("✅ Model loaded successfully!")
model = keras.models.load_model(MODEL_PATH)

# --- Title and Logo ---
st.image("aurora_logo.png", width=130)
st.title("Aurora Emotion Classifier")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a face image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Display image
    st.image(tmp_path, caption="Uploaded Image", width=300)

    # --- Prediction function ---
    def predict_emotion(image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "Image Read Error"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))

        if len(faces) == 0:
            face = cv2.resize(gray, (48, 48))
        else:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        prediction = model.predict(face, verbose=0)
        emotion_names = ['Happy', 'Sad', 'Angry']
        return emotion_names[np.argmax(prediction)]

    # --- Predict Button ---
    if st.button("🔍 Classify Emotion"):
        with st.spinner("Analyzing image..."):
            emotion = predict_emotion(tmp_path)
        st.subheader(f"Detected Emotion: {emotion}")

        # Highlight result with style
        st.markdown(f"""
            <div style="text-align:center; font-size: 1.6rem; 
                        background-color: rgba(0,255,255,0.1);
                        border: 1px solid #00ffff;
                        border-radius: 15px;
                        padding: 10px;
                        margin-top: 15px;
                        color: #00ffff;">
                {emotion}
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("📤 Please upload an image to classify the emotion.")

