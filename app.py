import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import io
import pickle
import base64
import pandas as pd

# Load the trained model
MODEL_PATH = "trained_model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# JavaScript-based audio recording function
def audio_recorder():
    """JavaScript-based audio recording using HTML5."""
    st.write("Click the button below to record audio.")
    js_code = """
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
                mediaRecorder.onstop = () => {
                    let audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    let reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = () => {
                        let base64Audio = reader.result.split(',')[1];
                        let py_audio = document.getElementById("audio_data");
                        py_audio.value = base64Audio;
                        document.getElementById("submit_audio").click();
                    };
                };
            });
        }
        function stopRecording() {
            mediaRecorder.stop();
        }
    </script>
    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop Recording</button>
    <input type="hidden" id="audio_data" name="audio_data">
    <button id="submit_audio" style="display:none;" onclick="document.getElementById('audio_data').dispatchEvent(new Event('change'))">Submit</button>
    """
    st.components.v1.html(js_code, height=300)
    
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
    summary_features_per_turn = summary_features_per_turn.reshape(1, -1)  # Shape (1, 156)

    max_across_turns = np.max(summary_features_per_turn, axis=0)
    mean_across_turns = np.mean(summary_features_per_turn, axis=0)
    std_across_turns = np.std(summary_features_per_turn, axis=0)

    final_468_features = np.concatenate([max_across_turns, mean_across_turns, std_across_turns])  # Shape (468,)
    return final_468_features.reshape(1, -1)  # Reshape for model input

# Function to predict dementia probability
def predict_dementia(features):
    probabilities = model.predict_proba(features)[0]  # Get probabilities
    return probabilities

# Streamlit App UI
st.title("🎤 Dementia Prediction Using Audio")
st.write("Upload an audio file or record your voice to extract features and predict the probability of dementia.")

# Audio recording
audio_recorder()

# File uploader
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features_468 = extract_468_features(uploaded_file)

    st.write("### Extracted 468-Dimensional Features:")
    st.write(features_468)

    dementia_prob = predict_dementia(features_468)
    st.write(f"### Dementia Probability: {dementia_prob[1]*100:.2f}%")

# Handle recorded audio from JavaScript
recorded_audio = st.text_input("", key="audio_data")
if recorded_audio:
    audio_bytes = base64.b64decode(recorded_audio)
    audio_buffer = io.BytesIO(audio_bytes)
    st.audio(audio_buffer, format="audio/wav")

    features_468 = extract_468_features(audio_buffer)
    st.write("### Extracted 468-Dimensional Features:")
    st.write(features_468)

    dementia_prob = predict_dementia(features_468)
    st.write(f"### Dementia Probability: {dementia_prob[1]*100:.2f}%")

