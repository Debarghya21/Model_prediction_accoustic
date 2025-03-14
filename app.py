import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
import io
import pandas as pd
import pickle

# Load the trained model
MODEL_PATH = "trained_model.pkl"
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Function to record audio
def record_audio(duration=5, samplerate=22050):
    st.write("Recording... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.float32)
    sd.wait()
    st.write("Recording complete!")

    buffer = io.BytesIO()
    sf.write(buffer, audio_data, samplerate, format='WAV')
    buffer.seek(0)
    
    return buffer

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

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if st.button("🎙️ Record Audio"):
    recorded_audio = record_audio(duration=5)
    if recorded_audio is None:
        st.error("No audio recorded. Please try again.")
    else:   
        st.audio(recorded_audio, format="audio/wav")
        features_468 = extract_468_features(recorded_audio)

        st.write("### Extracted 468-Dimensional Features:")
        st.write(features_468)

        dementia_prob = predict_dementia(features_468)
        st.write(f"### Dementia Probability: {dementia_prob[1]*100:.2f}%")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features_468 = extract_468_features(uploaded_file)

    st.write("### Extracted 468-Dimensional Features:")
    st.write(features_468)

    dementia_prob = predict_dementia(features_468)
    st.write(f"### Dementia Probability: {dementia_prob[1]*100:.2f}%")
