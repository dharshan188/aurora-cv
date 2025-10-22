
import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from collections import deque

# Import your predictor (assumes predict_emotion(image_or_path) exists)
# predict_emotion should accept either a file path (str) or a NumPy BGR image array
try:
    from predict_emotion import predict_emotion
except Exception as e:
    predict_emotion = None
    st = st  # keep linter happy
    print("Warning: could not import predict_emotion. Error:", e)

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="🧭",
    layout="centered",
)

# ---------------- Custom CSS - clean modern look ----------------
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #071029 50%, #071029 100%);
        color: #e6eef8;
        font-family: 'Inter', sans-serif;
    }

    /* container glass */
    .container {
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }

    h1 {
        margin-bottom: 6px;
        color: #e6eef8;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .subtitle {
        color: #9fb0c9;
        margin-top: 0;
        margin-bottom: 18px;
    }

    .result-card {
        display: flex;
        gap: 20px;
        align-items: center;
        justify-content: center;
        padding: 18px;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
        font-size: 1.4rem;
        font-weight: 700;
        color: #e6eef8;
    }

    .big-result {
        font-size: 2.2rem;
        letter-spacing: 0.6px;
    }

    .history {
        color: #9fb0c9;
        font-size: 0.95rem;
    }

    /* buttons */
    .stButton>button {
        background: linear-gradient(90deg,#2b6cb0,#0ea5a5);
        color: white;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown('<div class="container">', unsafe_allow_html=True)
st.markdown("<h1>Emotion Classifier</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clean, simple emotion detection — outputs only: Happy, Sad, Angry.</div>', unsafe_allow_html=True)

# ---------------- Layout: sidebar options ----------------
with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Choose input", ("Upload Image", "Use Webcam"))
    st.markdown("---")
    st.write("Options")
    show_history = st.checkbox("Show recent results", value=True)
    st.markdown("---")
    st.write("Tip: For best results, use a clear face photo (frontal).")

# create a small in-memory history for the session
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=8)

# ---------------- Input handling ----------------
uploaded_image = None
tmp_path = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload JPG / PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # save to temp file and also decode for display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        # decode for display
        file_bytes = np.asarray(bytearray(open(tmp_path, "rb").read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(uploaded_image, channels="BGR", caption="Uploaded Image", use_column_width=False, clamp=True)

else:  # Webcam
    cam_img = st.camera_input("Capture from webcam")
    if cam_img is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(cam_img.getvalue())
            tmp_path = tmp.name
        file_bytes = np.asarray(bytearray(open(tmp_path, "rb").read()), dtype=np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(uploaded_image, channels="BGR", caption="Captured Image", use_column_width=False, clamp=True)

# ---------------- Predict area ----------------
col1, col2 = st.columns([1, 1])
with col1:
    if uploaded_image is None:
        st.markdown("**Image ready:** No")
        btn_disabled = True
    else:
        st.markdown("**Image ready:** Yes")
        btn_disabled = False

with col2:
    if st.button("Classify Emotion", disabled=btn_disabled):
        if uploaded_image is None:
            st.warning("Please upload or capture an image first.")
        else:
            # call predict_emotion safely
            try:
                # if predict_emotion expects file path, pass tmp_path; if expects array, pass image
                # We'll try array first, then fallback to path.
                if predict_emotion is None:
                    raise RuntimeError("predict_emotion function not available (import failed).")
                # Try with array
                try:
                    raw_result = predict_emotion(uploaded_image)
                except Exception:
                    # fallback to path if we have one
                    if tmp_path:
                        raw_result = predict_emotion(tmp_path)
                    else:
                        raise

                # Clean output: map to only allowed labels (case-insensitive)
                allowed = ["Happy", "Sad", "Angry"]
                # raw_result may contain extra text like emoji or percentage; extract first token that's an allowed label
                result_label = None
                if isinstance(raw_result, str):
                    # Normalize and search for allowed labels
                    for token in raw_result.replace(",", " ").split():
                        t = token.strip().capitalize()
                        if t in allowed:
                            result_label = t
                            break
                if result_label is None:
                    # as a fallback, take the top token
                    result_label = str(raw_result).split()[0].capitalize()

                # store in session history
                st.session_state.history.appendleft(result_label)
                # show big result
                st.markdown(f'<div class="result-card"><div class="big-result">{result_label}</div></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error("Error while predicting: " + str(e))

# ---------------- History ----------------
if show_history:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='history'><b>Recent results (this session):</b></div>", unsafe_allow_html=True)
    if len(st.session_state.history) == 0:
        st.markdown("<div class='history'>No results yet.</div>", unsafe_allow_html=True)
    else:
        hist_html = "<div class='history'>"
        for idx, v in enumerate(st.session_state.history):
            hist_html += f"{idx+1}. {v}<br>"
        hist_html += "</div>"
        st.markdown(hist_html, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
