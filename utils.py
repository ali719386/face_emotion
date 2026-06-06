# utils.py
import soundfile as sf
import numpy as np

def save_uploaded_audio(uploaded_file, out_path="temp_audio.wav"):
    # uploaded_file is a bytes-like from streamlit.file_uploader
    data, samplerate = sf.read(uploaded_file)
    sf.write(out_path, data, samplerate)
    return out_path
