import torch
import torchaudio
from pathlib import Path
import pandas as pd
import joblib
import argparse
from collections import Counter, defaultdict
import editdistance
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import random


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop_size = max(frame_size_ms * sample_rate // 1000, 1)
    return int((timestamp * sample_rate) // hop_size)


def cut_encoding(
    encoding: torch.Tensor, word_boundaries: list[float], hop_ms: int = 20
) -> torch.Tensor:
    hop_size = hop_ms / 1000  # seconds per frame
    start_frame = int(word_boundaries[0] / hop_size)
    end_frame = int(word_boundaries[1] / hop_size)
    return encoding[start_frame:end_frame]


def collapse_runs(seq):
    collapsed = []
    prev = None
    for code in seq:
        if code != prev:
            collapsed.append(int(code))
            prev = code
    return collapsed


def middle_k(seq, k=3):
    n = len(seq)
    start = max((n - k) // 2, 0)
    return seq[start : start + k]


def top_k_codes(seq, k=3):
    return [int(code) for code, _ in Counter(seq).most_common(k)]


def top_k_codes_ordered_by_first_occurrence(seq, k=3):
    counts = Counter(seq)
    seen = set()
    result = []

    for code in seq:
        if code in seen:
            continue
        if counts[code] > 0:
            result.append(int(code))
            seen.add(code)
        if len(result) == k:
            break

    return result


def distance(feat_1, feat_2):
    length = max(len(feat_1), len(feat_2))
    return editdistance.eval(feat_1, feat_2) / length


def main(model_name: str, layer: int):
    sr = 16000
    align_dir = Path("librispeech/alignments")
    align_df = pd.read_csv(align_dir / "alignments.csv")

    # Load model
    try:
        bundle = getattr(torchaudio.pipelines, model_name)
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name}")

    model = bundle.get_model()
    model.eval()

    # Load a test file and alignment
    audio_dir = Path("librispeech/audio")
    audio_ext = ".flac"

    audio_paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    paths = random.sample(audio_paths, 10)
    # Load KMeans model
    kmeans_path = f"models/kmeans_{model_name.lower()}_layer{layer}_k100.pkl"
    kmeans = joblib.load(kmeans_path)

    summary_vectors = []
    for path in paths:
        wav_df = align_df[align_df["filename"] == path.stem]
        waveform, sr_loaded = torchaudio.load(path)
        if sr_loaded != sr:
            waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer)
            encoding = features[layer - 1].squeeze().cpu().numpy()

        # Extract encodings for "lady" instances
        labels = []
        cut_encodings = []
        for w in range(1, max(wav_df["word_id"]) + 1):
            word_df = wav_df[wav_df["word_id"] == w]
            labels.append(word_df["text"].iloc[0])
            cut_enc = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            cut_encodings.append(cut_enc)

        for w, enc in enumerate(cut_encodings, 1):
            word_codes = kmeans.predict(enc)
            summary = collapse_runs(word_codes)

            summary_vectors.append(summary)

    distance_matrix = np.zeros((len(summary_vectors), len(summary_vectors)))
    for i in range(len(summary_vectors)):
        for j in range(i, len(summary_vectors)):
            distance_matrix[i, j] = distance(summary_vectors[i], summary_vectors[j])

    distance_matrix += distance_matrix.T
    dist_df = pd.DataFrame(distance_matrix, index=labels, columns=labels)

    sns.clustermap(
        dist_df,
        metric="euclidean",  # clustering distance (already computed, so doesn't matter much here)
        method="average",  # linkage method
        cmap="Blues",
        figsize=(10, 10),
        annot=True,  # show numbers in grid
        fmt=".2f",
        cbar_kws={"label": "Distance"},
    )
    plt.title("Clustered Distance Heatmap")  # this won't show directly on clustermap
    plt.show()
    clustering = AgglomerativeClustering(n_clusters=40, linkage="average")
    cluster_labels = clustering.fit_predict(distance_matrix)

    cluster_groups = defaultdict(list)
    for label, word in zip(cluster_labels, labels):
        cluster_groups[label].append(word)

    # Print each cluster group
    print("\nCluster assignments (grouped by cluster):")
    for cluster_id in sorted(cluster_groups.keys()):
        print(f"\n🔹 Cluster {cluster_id}:")
        for word in cluster_groups[cluster_id]:
            print(f"  - {word}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )
    parser.add_argument(
        "--layer", type=int, default=8, help="Layer number to extract features from"
    )
    args = parser.parse_args()
    main(args.model, args.layer)
