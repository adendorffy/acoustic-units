import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

INT16_MAX = (2**15) - 1
HOP_LENGTH = 320
SAMPLE_RATE = 16000


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop_size = max(frame_size_ms * sample_rate // 1000, 1)
    return int((timestamp * sample_rate) // hop_size)


def cut_encoding(encoding: torch.Tensor, word_boundaries: list[float]) -> torch.Tensor:
    start_frame = get_frame_num(word_boundaries[0], SAMPLE_RATE, 20)
    end_frame = get_frame_num(word_boundaries[1], SAMPLE_RATE, 20)

    return encoding[start_frame:end_frame]


def get_units(
    gamma: float,
    layer: int,
    audio_dir: Path,
    align_dir: Path,
    feat_dir: Path,
    audio_ext: str = ".flac",
):
    paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    total_files = len(paths)

    if total_files == 0:
        print("⚠️ No audio files found.")
        return

    print(f"Processing {total_files} audio files...", flush=True)

    align_df = pd.read_csv(align_dir / "alignments.csv")
    features_dir = feat_dir / str(gamma) / str(layer)
    feat_paths = list(features_dir.rglob("**/*.npy"))

    total_words = len(align_df["filename"])
    print(f"Total words: {total_words}", flush=True)
    if len(feat_paths) == total_words:
        print(f"Already encoded audio using gamma = {gamma} at layer {layer}")
        return
    elif len(feat_paths) > 0:
        print(f"Warning: {len(feat_paths)} already encoded.")

    kmeans, segment = torch.hub.load(
        "bshall/dusted:main", "kmeans", language="english", trust_repo=True
    )
    hubert, encode = torch.hub.load(
        "bshall/dusted:main", "hubert", language="english", trust_repo=True
    )
    word_count = 0
    prev_progress = -1
    for i, path in enumerate(paths, start=1):
        wav_df = align_df[align_df["filename"] == path.stem]

        wav, sr = torchaudio.load(str(path))
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.unsqueeze(0)

        encoding = encode(hubert, wav, layer).squeeze(0)
        word_count += max(wav_df["word_id"]) + 1

        for w in range(max(wav_df["word_id"]) + 1):
            word_df = wav_df[wav_df["word_id"] == w]
            clean_encoding = cut_encoding(
                encoding,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )

            codes = []
            if clean_encoding.numel() > 0:
                codes, _ = segment(
                    clean_encoding.numpy(), kmeans.cluster_centers_, gamma
                )

            save_path = (
                features_dir
                / path.relative_to(audio_dir).parent
                / f"{path.stem}_{w}.npy"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, codes)

            progress = int((i / total_files) * 100)
            if progress % 10 == 0 and progress > prev_progress:
                print(f"🟢 Progress: {progress}% ({i}/{total_files} files)", flush=True)
                prev_progress = progress

    print("\n✅ Processing complete!")
    if word_count != total_words:
        print(f"⚠️ Warning: Processed {word_count} words, but expected {total_words}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio files and extract speech units."
    )
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "audio_dir", type=Path, help="Path to the directory containing audio files."
    )
    parser.add_argument(
        "align_dir", type=Path, help="Path to the directory with the alignments."
    )
    parser.add_argument(
        "feat_dir", type=Path, help="Path to the directory to store encodings."
    )

    parser.add_argument(
        "--audio_ext",
        type=str,
        default=".flac",
        help="Audio file extension (default: .flac)",
    )

    args = parser.parse_args()

    get_units(
        args.gamma,
        args.layer,
        args.audio_dir,
        args.align_dir,
        args.feat_dir,
        args.audio_ext,
    )
