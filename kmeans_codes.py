import torch
import torchaudio
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

import numpy as np
from collections import Counter

sr = 16000
audio_dir = Path("librispeech/audio")
align_dir = Path("librispeech/alignments")
feat_dir = Path("kmeans-features/")
audio_ext = ".flac"
layer = 7


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop_size = max(frame_size_ms * sample_rate // 1000, 1)
    return int((timestamp * sample_rate) // hop_size)


def cut_encoding(waveform: torch.Tensor, word_boundaries: list[float]) -> torch.Tensor:
    start_frame = get_frame_num(word_boundaries[0], sr, 20)
    end_frame = get_frame_num(word_boundaries[1], sr, 20)
    return waveform[start_frame:end_frame]


def characterize_codes(codes, top_k=5, min_run_length=3):
    """
    Takes a 1D array of codes and returns a list of codes that characterize the array
    based on frequency and long runs.

    Args:
        codes (array-like): The input sequence of codes.
        top_k (int): Number of most frequent codes to return.
        min_run_length (int): Minimum run length to consider a code persistent.

    Returns:
        dict with:
            - 'top_k_frequent': top-k most frequent codes.
            - 'long_runs': codes with at least one run >= min_run_length.
            - 'combined': union of both.
    """
    codes = np.array(codes)

    # 1. Top-k frequent codes
    freq_counts = Counter(codes)
    top_k_frequent = [c for c, _ in freq_counts.most_common(top_k)]

    # 2. Codes with long runs
    long_runs = set()
    current_code = codes[0]
    run_length = 1
    for i in range(1, len(codes)):
        if codes[i] == current_code:
            run_length += 1
        else:
            if run_length >= min_run_length:
                long_runs.add(current_code)
            current_code = codes[i]
            run_length = 1
    # Don't forget to check the final run
    if run_length >= min_run_length:
        long_runs.add(current_code)

    combined = set(top_k_frequent).union(long_runs)

    return {
        "top_k_frequent": top_k_frequent,
        "long_runs": list(long_runs),
        "combined": list(combined),
    }


# Load alignment and model
align_df = pd.read_csv(align_dir / "alignments.csv")
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()
model.eval()

# Load audio
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


# Fit KMeans
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(encoding)


codes_per_word = []
for enc in cut_encodings:
    word_codes = kmeans.predict(enc)
    codes_per_word.append(word_codes)
    print(word_codes)
    # character = characterize_codes(word_codes)
    # for inf in character:
    #     print(inf, "-".join([str(el) for el in character[inf]]))
    # print()
