import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import pickle
import io
from io import BytesIO
import base64

# Load the trained model
MODEL_PATH = "trained_model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Function to extract 468-dimensional feature vector
def extract_468_features(audio_file, sr=22050, n_mfcc=13):
    y, sr = librosa.load(audio_file, sr=sr)
    window_length = int(0.023 * sr)  # 23ms window
    hop_length = int(0.010 * sr)  # 10ms step size

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=window_length, window='hamming')
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    features_per_frame = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    min_features = np.min(features_per_frame, axis=1)
    max_features = np.max(features_per_frame, axis=1)
    mean_features = np.mean(features_per_frame, axis=1)
    std_features = np.std(features_per_frame, axis=1)

    summary_features_per_turn = np.hstack([min_features, max_features, mean_features, std_features])
    summary_features_per_turn = summary_features_per_turn.reshape(1, -1)

    max_across_turns = np.max(summary_features_per_turn, axis=0)
    mean_across_turns = np.mean(summary_features_per_turn, axis=0)
    std_across_turns = np.std(summary_features_per_turn, axis=0)

    final_468_features = np.concatenate([max_across_turns, mean_across_turns, std_across_turns])
    return final_468_features.reshape(1, -1)

# Function to predict dementia probability
def predict_dementia(features):
    probabilities = model.predict_proba(features)[0]
    return probabilities

# Streamlit UI
st.title("🎤 Dementia Prediction Using Audio")
st.write("Upload an audio file or record your voice to extract features and predict the probability of dementia.")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

# JavaScript to record audio
st.write("Or record your voice below:")
record_button = st.button("🎙️ Start Recording")

record_script = """
<script>
let mediaRecorder;
let audioChunks = [];

function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            let blob = new Blob(audioChunks, { type: 'audio/wav' });
            let reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = () => {
                let base64data = reader.result.split(',')[1];
                fetch('/send_audio', {
                    method: 'POST',
                    body: JSON.stringify({ audio: base64data }),
                    headers: { 'Content-Type': 'application/json' }
                });
            };
        };
    });
}

document.querySelector("button").addEventListener("click", startRecording);
</script>
"""
st.markdown(record_script, unsafe_allow_html=True)

# Process uploaded file
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features_468 = extract_468_features(uploaded_file)
    st.write("### Extracted 468-Dimensional Features:")
    st.write(features_468)
    dementia_prob = predict_dementia(features_468)
    st.write(f"### Dementia Probability: {dementia_prob[1] * 100:.2f}%")

