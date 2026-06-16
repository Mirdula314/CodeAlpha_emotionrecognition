import streamlit as st
import librosa
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import tempfile

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎙️",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("emotion_model.pkl")

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(
        135deg,
        #2A193D 0%,
        #2F2C5E 45%,
        #6D4B6C 100%
    );
}

/* Hide Streamlit header */
[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Hero section */
.hero {
    background: linear-gradient(
        90deg,
        #B5D4D9,
        #7F5DA3
    );

    padding: 35px;
    border-radius: 25px;
    color: #2A193D;
    margin-bottom: 25px;

    box-shadow:
    0px 10px 30px rgba(0,0,0,0.25);
}

/* Headings */
h1,h2,h3,h4,label,p {
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 18px;
    padding: 15px;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(
        90deg,
        #7F5DA3,
        #6D4B6C
    );
    color: white;
    border: none;
    border-radius: 14px;
    height: 55px;
    font-size: 18px;
    font-weight: 600;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 20px;
}

/* Metric text */
[data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetricLabel"] {
    color: #B5D4D9 !important;
}

/* Success box */
.stSuccess {
    border-radius: 15px;
}

/* Info box */
.stInfo {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class='hero'>

<h1 style="
font-size:50px;
margin-bottom:10px;
color:#2A193D;
">

🎙️ Emotion Recognition System

</h1>

<p style="
font-size:20px;
color:#2A193D;
">

Analyze human emotions from speech using Machine Learning and Audio Signal Processing.

</p>

</div>
""", unsafe_allow_html=True)

# =====================================
# FEATURE EXTRACTION
# =====================================

def extract_features(file_path):
    
    audio, sr = librosa.load(
        file_path,
        duration=3,
        offset=0.5
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)

# =====================================
# LAYOUT
# =====================================

left, right = st.columns([1, 1])

with left:

    st.subheader("🎵 Upload Audio")

    uploaded_file = st.file_uploader(
        "Choose a WAV File",
        type=["wav"]
    )

with right:

    st.subheader("📋 Project Information")

    st.info(
        """
        This AI system analyzes speech recordings and predicts
        human emotions using audio feature extraction and
        machine learning techniques.
        """
    )

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze Emotion"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(uploaded_file.read())
            file_path = tmp.name

        features = extract_features(file_path)
        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = np.max(probabilities) * 100

        st.markdown("---")

        st.subheader("🎯 Analysis Result")

        emotion_icons = {
            "Happy": "😊",
            "Sad": "😔",
            "Angry": "😠",
            "Fearful": "😨",
            "Neutral": "😐",
            "Calm": "😌",
            "Disgust": "🤢",
            "Surprised": "😲"
        }

        icon = emotion_icons.get(prediction, "🎭")

        st.success(
            f"{icon} Predicted Emotion: {prediction}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Emotion",
                prediction
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        # =====================================
        # CONFIDENCE GAUGE
        # =====================================

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            title={"text": "Confidence Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#7F5DA3"},
                "steps": [
                    {"range": [0, 30], "color": "#B5D4D9"},
                    {"range": [30, 70], "color": "#7F5DA3"},
                    {"range": [70, 100], "color": "#6D4B6C"}
                ]
            }
        ))

        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )

        st.plotly_chart(
            gauge_fig,
            use_container_width=True
        )

        # =====================================
        # PROBABILITY CHART
        # =====================================

        emotions = model.classes_

        chart_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probabilities
        })

        prob_fig = go.Figure()

        prob_fig.add_bar(
            x=chart_df["Emotion"],
            y=chart_df["Probability"],
            marker_color="#7F5DA3"
        )

        prob_fig.update_layout(
            title="Emotion Probability Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis_title="Emotion",
            yaxis_title="Probability"
        )

        st.plotly_chart(
            prob_fig,
            use_container_width=True
        )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.info(
    "This application predicts emotions from speech using Machine Learning, MFCC features, Chroma features, and Mel Spectrogram analysis."
)