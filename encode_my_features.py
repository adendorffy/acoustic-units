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
    """Group speech representations into phone-like segments.

    Args:
        sequence (NDArray): speech representations of shape (T, D) where T is the number of frames and D is the feature dimension.
        codebook (NDArray): cluster centriods of the discrete units of shape (K, D) where K is the number of codes.
        gamma float: Duration regularizer weight. Larger values result in a coarser segmentation.

    Returns:
        NDArray[int]: list of discrete units representing each segment sound types of shape (N,).
        NDArray[int]: list of segment boundaries of shape (N+1,).
    """
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
    hop_size = hop_ms / 1000  # seconds per frame
    start_frame = int(word_boundaries[0] / hop_size)
    end_frame = int(word_boundaries[1] / hop_size)
    return encoding[start_frame:end_frame]


def main(model_name: str, layer: int, gamma: float):
    model_name = model_name.upper()
    sr = 16000
    align_dir = Path("librispeech/alignments")
    align_df = pd.read_csv(align_dir / "alignments.csv")
    features_dir = Path(f"my-features/{model_name}/{layer}/gamma{gamma}")
    features_dir.mkdir(parents=True, exist_ok=True)

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

    kmeans_path = f"models/kmeans_{model_name.lower()}_layer{layer}_k100.pkl"
    kmeans = joblib.load(kmeans_path)
    align_df["codes"] = None

    for path in tqdm(
        audio_paths,
        total=len(audio_paths),
        desc="Encoding features",
    ):
        wav_df = align_df[align_df["filename"] == path.stem]
        waveform, sr_loaded = torchaudio.load(str(path))
        if sr_loaded != sr:
            waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer)
            encoding = features[layer - 1].squeeze().cpu().numpy()

        for w in range(1, max(wav_df["word_id"]) + 1):
            word_df = wav_df[wav_df["word_id"] == w]

            word_encoding = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            codes = []
            if len(word_encoding) > 0:
                codes = segment(word_encoding, kmeans.cluster_centers_, gamma)

            save_path = (
                features_dir
                / path.relative_to(audio_dir).parent
                / f"{path.stem}_{w}.npy"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, codes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )
    parser.add_argument("gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument(
        "layer", type=int, default=8, help="Layer number to extract features from"
    )

    args = parser.parse_args()
    main(args.model, args.layer, args.gamma)
