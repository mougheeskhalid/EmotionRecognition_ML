# Emotion Recognition from Speech

Classifies the emotional state of a speaker (neutral, calm, happy, sad,
angry, fearful, disgust, surprised) from raw speech audio using MFCC
features and a CNN-LSTM network.

## Approach

- Audio is loaded with `librosa`, trimmed to a fixed 4-second window, and
  converted into 40 MFCCs (Mel-Frequency Cepstral Coefficients), padded or
  truncated to a fixed length so every sample has the same shape.
- Extracted features are cached to `data/features_cache.npz` so re-running
  the script doesn't re-process every audio file.
- The model combines 1D convolutions (to pick up local spectral patterns)
  with stacked LSTM layers (to capture how those patterns evolve over
  time), followed by a dense classification head.
- Training uses early stopping and learning-rate reduction on plateau.

## Dataset

This script targets the **RAVDESS** (Ryerson Audio-Visual Database of
Emotional Speech and Song) dataset. It also works with **TESS** or
**EMO-DB** if you adapt `parse_emotion_from_filename` to their naming
convention — the RAVDESS parser is provided as the default since it's the
most commonly used of the three.

1. Download the "Audio_Speech_Actors_01-24" set from:
   https://zenodo.org/record/1188976
2. Extract it so you get `data/Actor_01/`, `data/Actor_02/`, ... each
   containing `.wav` files named like `03-01-06-01-02-01-12.wav`.

RAVDESS filenames encode metadata positionally:
`modality-vocalChannel-emotion-intensity-statement-repetition-actor`.
The 3rd field is the emotion code (01=neutral … 08=surprised), which is
what `parse_emotion_from_filename` reads.

## Project structure

```
Task2_EmotionRecognition_Speech/
├── emotion_recognition.py
├── requirements.txt
├── data/       # place the RAVDESS Actor_* folders here
├── models/     # saved .keras model (created on run)
└── outputs/    # training curves + confusion matrix (created on run)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python emotion_recognition.py --data-dir data --epochs 40
```

Optional flags: `--batch-size` (default 32), `--cache` (path to the
feature cache, default `data/features_cache.npz`). Delete the cache file
if you change the dataset and want features re-extracted.

## Output

- `models/speech_emotion_cnn_lstm.keras` — the trained model
- `outputs/training_history.png` — accuracy/loss curves
- `outputs/confusion_matrix.png` — per-emotion confusion matrix
- A classification report printed to the console

Expect roughly 55–65% test accuracy on RAVDESS's 8-way classification with
this architecture and no data augmentation — a reasonable baseline that
can be improved with augmentation (pitch shift, time stretch, noise
injection) or a pretrained audio embedding model (e.g. Wav2Vec2).
