# train_voice_model.py  —  IMPROVED VERSION (fixes overfitting)
# Run: python train_voice_model.py
# Dataset already downloaded — will skip download and retrain only.

import os
import zipfile
import requests
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_DIR = "ravdess_data"
MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "voice_model.h5")
N_MFCC      = 40
MAX_PAD_LEN = 174
SR          = 22050

RAVDESS_SPEECH_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

LABEL_LIST   = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise", "anxious"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_LIST)}


# ── Download with resume ──────────────────────────────────────────────────────
def download_ravdess():
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path    = os.path.join(DATASET_DIR, "ravdess.zip")
    extract_dir = os.path.join(DATASET_DIR, "extracted")

    if not os.path.exists(extract_dir):
        existing = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
        headers  = {"Range": f"bytes={existing}-"} if existing else {}
        print(f"[INFO] Downloading RAVDESS (resuming from {existing // 1024 // 1024} MB)...")
        for attempt in range(1, 11):
            try:
                with requests.get(RAVDESS_SPEECH_URL, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("content-length", 0)) + existing
                    with open(zip_path, "ab" if existing else "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            f.write(chunk)
                            existing += len(chunk)
                            if total:
                                print(f"\r  {existing/total*100:.1f}%", end="", flush=True)
                print("\n[INFO] Download complete.")
                break
            except Exception as e:
                existing = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
                headers  = {"Range": f"bytes={existing}-"}
                print(f"\n[WARN] Attempt {attempt}/10 failed. Retrying...")
                if attempt == 10:
                    raise
        print("[INFO] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("[INFO] Extraction complete.")
    else:
        print("[INFO] Dataset already extracted — skipping download.")
    return extract_dir


# ── Feature extraction with augmentation ─────────────────────────────────────
def extract_mfcc(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_PAD_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_PAD_LEN - mfcc.shape[1])), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc


def augment_audio(y, sr):
    """Return list of augmented versions of the audio."""
    augmented = [y]  # original

    # 1. Add white noise
    noise = np.random.randn(len(y)) * 0.005
    augmented.append(y + noise)

    # 2. Time stretch
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    except Exception:
        pass

    # 3. Pitch shift
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except Exception:
        pass

    return augmented


def load_dataset(extract_dir, augment=True):
    X, y = [], []
    wav_files = []
    for root, _, files in os.walk(extract_dir):
        for fname in files:
            if fname.endswith(".wav"):
                wav_files.append(os.path.join(root, fname))

    print(f"[INFO] Found {len(wav_files)} wav files.")
    for fpath in wav_files:
        parts = os.path.basename(fpath).replace(".wav", "").split("-")
        if len(parts) < 3:
            continue
        label = RAVDESS_EMOTION_MAP.get(parts[2])
        if label is None:
            continue
        idx = LABEL_TO_IDX[label]

        try:
            audio, sr = librosa.load(fpath, sr=SR)
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")
            continue

        versions = augment_audio(audio, sr) if augment else [audio]
        for version in versions:
            mfcc = extract_mfcc(version, sr)
            X.append(mfcc)
            y.append(idx)

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Loaded {len(X)} samples (with augmentation) across {len(set(y))} classes.")
    return X, y


# ── Model (lighter to reduce overfitting) ────────────────────────────────────
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),

        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Voice Emotion Model Training  (improved)")
    print("=" * 60)

    extract_dir = download_ravdess()

    print("[INFO] Extracting features + augmenting (10-15 min)...")
    X, y = load_dataset(extract_dir, augment=True)

    if len(X) == 0:
        print("[ERROR] No data loaded.")
        return

    X     = X[..., np.newaxis]
    y_cat = to_categorical(y, num_classes=len(LABEL_LIST))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)}, Val: {len(X_val)}")

    model = build_model(X_train.shape[1:], len(LABEL_LIST))
    model.summary()

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(patience=6, factor=0.5, min_lr=1e-6, verbose=1),
    ]

    print("[INFO] Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n[SUCCESS] Model saved to {MODEL_PATH}")
    print(f"[INFO] Label order: {LABEL_LIST}")


if __name__ == "__main__":
    main()
