import torch
import torchaudio
from pathlib import Path
import pandas as pd
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple
import numba
import scipy.spatial.distance as distance


def segment(
    sequence: np.ndarray, codebook: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    dists = distance.cdist(sequence, codebook).astype(np.float32)
    alpha, P = _segment(dists, gamma)
    return _backtrack(alpha, P)


@numba.njit()
def _segment(dists, gamma):
    T, K = dists.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = dists[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + dists[s, :] - gamma

    for t in range(T):
        alpha[t + 1] = np.inf
        for s in range(t + 1):
            k = np.argmin(D[s, t, :])
            alpha_min = alpha[s] + D[s, t, k]
            if alpha_min < alpha[t + 1]:
                P[t + 1, :] = s, k
                alpha[t + 1] = alpha_min
    return alpha, P


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = []
    boundaries = [rhs]
    while rhs != 0:
        lhs, code = P[rhs, :]
        segments.append(code)
        boundaries.append(lhs)
        rhs = lhs
    segments.reverse()
    boundaries.reverse()
    return segments


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop_size = max(frame_size_ms * sample_rate // 1000, 1)
    return int((timestamp * sample_rate) // hop_size)


def cut_encoding(
    encoding: torch.Tensor, word_boundaries: list[float], hop_ms: int = 20
) -> torch.Tensor:
    hop_size = hop_ms / 1000
    start_frame = int(word_boundaries[0] / hop_size)
    end_frame = int(word_boundaries[1] / hop_size)
    return encoding[start_frame:end_frame]


def main(
    audio_dir: Path,
    align_dir: Path,
    model_name: str,
    layer: int,
    gamma: float,
    n_clusters: int,
):
    align_df = pd.read_csv(align_dir / "alignments.csv")

    raw_features_dir = Path("raw_features") / audio_dir / model_name / f"layer{layer}"
    raw_paths = list(raw_features_dir.rglob("**/*.npy"))

    if len(raw_paths) == 0:
        print(
            f"No raw features found in {raw_features_dir}. Please run the feature extraction script first.",
            flush=True,
        )
        return

    print(f"Encoding {len(raw_paths)} audio files from {raw_features_dir}", flush=True)

    try:
        bundle = getattr(torchaudio.pipelines, model_name.upper())
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name.upper()}")

    model = bundle.get_model()
    model.eval()

    features_dir = Path(
        f"features/{model_name}/layer{layer}/gamma{gamma}/k{n_clusters}"
    )
    features_dir.mkdir(parents=True, exist_ok=True)
    feat_paths = list(features_dir.rglob("**/*.npy"))

    if len(feat_paths) == len(raw_paths):
        print(
            f"All {len(feat_paths)} features already encoded in {features_dir}. Skipping encoding.",
            flush=True,
        )
        return
    kmeans_path = f"kmeans_models/kmeans_{model_name}_layer{layer}_k{n_clusters}.pkl"
    kmeans = joblib.load(kmeans_path)
    print(
        f"Loaded KMeans model from {kmeans_path} to encode {len(feat_paths)} features.",
        flush=True,
    )

    paths = []
    for path in tqdm(
        raw_paths,
        total=len(raw_paths),
        desc="Encoding features",
    ):
        wav_df = align_df[align_df["filename"] == path.stem]
        encoding = np.load(path)
        if wav_df.empty:
            print(f"No alignment found for {path.stem}. Skipping.", flush=True)
            continue

        for w in range(1, max(wav_df["word_id"]) + 1):
            word_df = wav_df[wav_df["word_id"] == w]
            save_path = (
                features_dir
                / path.relative_to(raw_features_dir).parent
                / f"{path.stem}_{w}.npy"
            )
            if save_path in feat_paths:
                continue

            word_encoding = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            codes = []
            if len(word_encoding) > 0:
                codes = segment(word_encoding, kmeans.cluster_centers_, gamma)

            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, codes)

            paths.append(f"{path.stem}_{w}")

    paths_df = pd.DataFrame([p for p in paths])
    paths_df.to_csv(features_dir / "paths.csv", index=True, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        type=Path,
        default="librispeech/dev-clean",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "align_dir",
        type=Path,
        help="Path to the directory containing alignments",
    )
    parser.add_argument(
        "model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )

    parser.add_argument(
        "layer", type=int, default=8, help="Layer number to extract features from"
    )
    parser.add_argument("gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument(
        "n_clusters", type=int, default=100, help="Number of clusters for KMeans"
    )

    args = parser.parse_args()
    print(
        f"encode_features.py, [{args.audio_dir}, {args.align_dir}, {args.model}, {args.layer}, {args.gamma}, {args.n_clusters}]",
        flush=True,
    )
    main(
        args.audio_dir,
        args.align_dir,
        args.model,
        args.layer,
        args.gamma,
        args.n_clusters,
    )
