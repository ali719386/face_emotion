import os
import numpy as np
import soundfile as sf
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("models", "voice_model.h5")


class VoiceEmotion:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            print(f"[VoiceEmotion] Warning: model not found at {model_path}. Predictions will be fallback neutral.")

        self.labels = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise", "anxious"]

    def _load_audio(self, file_path, sr=22050):
        """Load audio using soundfile first (fast), fallback to librosa for MP3 etc."""
        try:
            y, native_sr = sf.read(file_path, dtype='float32', always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)  # stereo → mono
            if native_sr != sr:
                y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
            return y, sr
        except Exception:
            # Fallback: librosa with ffmpeg backend (handles MP3, OGG, M4A etc.)
            y, sr = librosa.load(file_path, sr=sr)
            return y, sr

    def extract_features(self, file_path, sr=22050, n_mfcc=40, max_pad_len=174):
        y, sr = self._load_audio(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc

    def predict_from_file(self, file_path):
        mfcc = self.extract_features(file_path)
        X = np.expand_dims(mfcc, axis=(0, -1))
        if self.model:
            probs = self.model.predict(X, verbose=0)[0]
            label = self.labels[np.argmax(probs)]
            scores = {lab: float(probs[i]) for i, lab in enumerate(self.labels)}
            return {"label": label, "scores": scores}
        else:
            return {"label": "neutral", "scores": {"neutral": 1.0}}