import torch
import torchaudio
from pathlib import Path
import pandas as pd
import joblib
import argparse
import editdistance
import numpy as np
from tqdm import tqdm


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


def distance(feat_1, feat_2):
    length = max(len(feat_1), len(feat_2))
    return editdistance.eval(feat_1, feat_2) / length


def main(model_name: str, layer: int):
    sr = 16000
    align_dir = Path("librispeech/alignments")
    align_df = pd.read_csv(align_dir / "alignments.csv")
    features_dir = Path(f"my-features/{model_name}/{layer}")
    features_dir.mkdir(parents=True, exist_ok=True)

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

    # Load KMeans model
    kmeans_path = f"models/kmeans_{model_name.lower()}_layer{layer}_k100.pkl"
    kmeans = joblib.load(kmeans_path)

    word_count = 0
    for i, path in tqdm(
        enumerate(audio_paths, start=1),
        total=len(audio_paths),
        desc="Encoding features",
    ):
        wav_df = align_df[align_df["filename"] == path.stem]
        word_count += max(wav_df["word_id"]) + 1
        waveform, sr_loaded = torchaudio.load(path)
        if sr_loaded != sr:
            waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer)
            encoding = features[layer - 1].squeeze().cpu().numpy()

        # Extract encodings for "lady" instances
        for w in range(1, max(wav_df["word_id"]) + 1):
            word_df = wav_df[wav_df["word_id"] == w]
            cut_enc = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            codes = []
            if len(cut_enc) > 0:
                word_codes = kmeans.predict(cut_enc)
                codes = collapse_runs(word_codes)

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
    parser.add_argument(
        "layer", type=int, default=8, help="Layer number to extract features from"
    )
    args = parser.parse_args()
    main(args.model, args.layer)
