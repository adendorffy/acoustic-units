import torch
import torchaudio
from pathlib import Path
import pandas as pd
import joblib


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop_size = max(frame_size_ms * sample_rate // 1000, 1)
    return int((timestamp * sample_rate) // hop_size)


def cut_encoding(waveform: torch.Tensor, word_boundaries: list[float]) -> torch.Tensor:
    start_frame = get_frame_num(word_boundaries[0], sr, 20)
    end_frame = get_frame_num(word_boundaries[1], sr, 20)
    return waveform[start_frame:end_frame]


sr = 16000
audio_dir = Path("librispeech/audio")
align_dir = Path("librispeech/alignments")
feat_dir = Path("kmeans-features/")
audio_ext = ".flac"
layer = 8

align_df = pd.read_csv(align_dir / "alignments.csv")
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()
model.eval()

path = Path("librispeech/audio/174/50561/174-50561-0005.flac")
lady_indices = [2, 4, 9, 18]
wav_df = align_df[align_df["filename"] == path.stem]
waveform, sr = torchaudio.load(path)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

with torch.inference_mode():
    features, _ = model.extract_features(waveform, num_layers=layer)
    encoding = features[layer - 1].squeeze().cpu().numpy()

# Extract features per word
cut_encodings = []
for w in range(1, max(wav_df["word_id"]) + 1):
    if w in lady_indices:
        word_df = wav_df[wav_df["word_id"] == w]
        if word_df["text"].iloc[0] != "lady":
            print("Not a lady!!")
        cut_enc = cut_encoding(
            encoding,
            [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
        )
        cut_encodings.append(cut_enc)

kmeans = joblib.load("models/kmeans_hubert_layer8_k100.pkl")

codes_per_word = []
for enc in cut_encodings:
    word_codes = kmeans.predict(enc)
    codes_per_word.append(word_codes)
    print(word_codes)
