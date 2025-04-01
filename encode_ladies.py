import torch
import torchaudio
from pathlib import Path
import pandas as pd
import joblib
import argparse


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
    path = Path("librispeech/audio/174/50561/174-50561-0005.flac")
    lady_indices = [2, 4, 9, 18]
    wav_df = align_df[align_df["filename"] == path.stem]
    waveform, sr_loaded = torchaudio.load(path)
    if sr_loaded != sr:
        waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform, num_layers=layer)
        encoding = features[layer - 1].squeeze().cpu().numpy()

    # Extract encodings for "lady" instances
    cut_encodings = []
    for w in range(1, max(wav_df["word_id"]) + 1):
        if w in lady_indices:
            word_df = wav_df[wav_df["word_id"] == w]
            if word_df["text"].iloc[0] != "lady":
                print(
                    f"Word ID {w}: Not a lady!! Found '{word_df['text'].iloc[0]}' instead."
                )
                continue
            cut_enc = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            cut_encodings.append(cut_enc)

    # Load KMeans model
    kmeans_path = f"models/kmeans_{model_name.lower()}_layer{layer}_k100.pkl"
    kmeans = joblib.load(kmeans_path)

    # Predict cluster codes for each "lady" word
    codes_per_word = []
    for enc in cut_encodings:
        word_codes = kmeans.predict(enc)
        codes_per_word.append(word_codes)
        print(word_codes)


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
