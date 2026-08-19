"""
Emotion Recognition from Speech
---------------------------------
Extracts MFCC features from speech audio and trains a CNN-LSTM network to
classify the speaker's emotion.

Expects the RAVDESS dataset laid out as downloaded, e.g.:

    data/
      Actor_01/
        03-01-01-01-01-01-01.wav
        03-01-06-01-02-01-01.wav
        ...
      Actor_02/
        ...

RAVDESS filenames encode the emotion in the 3rd field, e.g. 03-01-06-01-02-01-12.wav
    modality-vocalChannel-emotion-intensity-statement-repetition-actor

Usage:
    python emotion_recognition.py --data-dir data --epochs 40
"""

import argparse
import glob
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

N_MFCC = 40
MAX_PAD_LEN = 174  # ~4 seconds at default hop length


def parse_emotion_from_filename(path):
    fname = os.path.basename(path)
    code = fname.split("-")[2]
    return RAVDESS_EMOTIONS.get(code)


def extract_features(file_path, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    signal, sr = librosa.load(file_path, sr=None, duration=4, offset=0.5)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)

    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs


def build_dataset(data_dir):
    files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
    if not files:
        raise SystemExit(
            f"No .wav files found under '{data_dir}'. "
            "Download RAVDESS and point --data-dir at the extracted folder."
        )

    X, y = [], []
    print(f"Found {len(files)} audio files. Extracting MFCC features...")
    for i, path in enumerate(files, 1):
        emotion = parse_emotion_from_filename(path)
        if emotion is None:
            continue
        try:
            features = extract_features(path)
        except Exception as exc:  # skip unreadable/corrupt files
            print(f"Skipping {path}: {exc}")
            continue
        X.append(features)
        y.append(emotion)
        if i % 100 == 0:
            print(f"  processed {i}/{len(files)}")

    X = np.array(X)
    y = np.array(y)
    return X, y


def build_model(input_shape, num_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(128, 5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.Conv1D(256, 5, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
            layers.Dropout(0.4),

            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Speech emotion recognition")
    parser.add_argument("--data-dir", default="data", help="Path to RAVDESS root folder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", default=os.path.join("data", "features_cache.npz"))
    args = parser.parse_args()

    if os.path.exists(args.cache):
        print(f"Loading cached features from {args.cache}")
        cache = np.load(args.cache, allow_pickle=True)
        X, y = cache["X"], cache["y"]
    else:
        X, y = build_dataset(args.data_dir)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        np.savez(args.cache, X=X, y=y)
        print(f"Cached extracted features to {args.cache}")

    print(f"Dataset shape: {X.shape}, labels: {sorted(set(y))}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = list(encoder.classes_)
    num_classes = len(class_names)
    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes)

    # Normalize features (per-coefficient) and transpose to (time, features) for Conv1D/LSTM
    X = (X - X.mean()) / (X.std() + 1e-8)
    X = np.transpose(X, (0, 2, 1))  # (samples, time_steps, n_mfcc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = build_model(X_train.shape[1:], num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    plot_training_history(history, os.path.join(OUTPUTS_DIR, "training_history.png"))
    plot_confusion_matrix(y_true, y_pred, class_names, os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))

    model_path = os.path.join(MODELS_DIR, "speech_emotion_cnn_lstm.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Plots saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
