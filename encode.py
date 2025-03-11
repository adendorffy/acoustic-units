import random
import struct
import numpy as np
import torch
from webrtcvad import Vad
from tqdm import tqdm
import torchaudio
from pathlib import Path
import pandas as pd
import argparse


def sample_files(audio_dir=None, audio_ext=None, feature_dir=None, sample_size=100):
    """
    Sample files from audio_dir or feature_dir.
    If sample_size = -1, sample all the files in the dir.
    """

    if audio_dir and audio_ext:
        paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    elif feature_dir:
        paths = list(feature_dir.rglob("*/*.npy"))

    if sample_size < len(paths) and sample_size > 0:
        paths = random.sample(paths, sample_size)

    return paths, len(paths)


def mark_sil(vad, wav):
    """
    Detect speech activity in an audio signal using a VAD system.
    """

    INT16_MAX = (2**15) - 1
    hop_length = 320
    sample_rate = 16000

    wav = torch.nn.functional.pad(wav, (40, 40))
    wav = wav[:, : wav.size(-1) - (wav.size(-1) % hop_length)]

    pcm = struct.pack(
        "%dh" % wav.size(-1),
        *(np.round(wav.squeeze().numpy() * INT16_MAX)).astype(np.int16),
    )

    flags = []
    for window_start in range(0, wav.size(-1), hop_length):
        window_end = window_start + hop_length
        flag = vad.is_speech(pcm[window_start * 2 : window_end * 2], sample_rate)
        flags.append(flag)

    return flags


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int) -> int:
    hop = frame_size_ms / 1000 * sample_rate
    hop_size = np.max([hop, 1])
    return int((timestamp * sample_rate) / hop_size)


def cut_and_clean_encoding(encoding, flags, word_boundaries):
    start_frame = get_frame_num(word_boundaries[0], 16000, 20)
    end_frame = get_frame_num(word_boundaries[1], 16000, 20)

    cut_encoding = encoding[start_frame:end_frame]
    cut_flags = flags[start_frame:end_frame]

    clean_encoding = []
    for i in range(min(cut_encoding.shape[0], len(flags))):
        if cut_flags[i]:
            clean_encoding.append(cut_encoding[i, :].unsqueeze(0))

    if clean_encoding != []:
        clean_encoding = torch.cat(clean_encoding, dim=0)
    return clean_encoding


def get_units(paths, align_df, wav_dir, gamma, layer, save_dir):
    kmeans, segment = torch.hub.load(
        "bshall/dusted:main", "kmeans", language="english", trust_repo=True
    )
    hubert, encode = torch.hub.load(
        "bshall/dusted:main", "hubert", language="english", trust_repo=True
    )
    vad = Vad()

    for path in tqdm(paths, desc="Getting units"):
        wav_df = align_df[align_df["filename"] == path.stem]

        wav, sr = torchaudio.load(str(path))
        wav = torchaudio.functional.resample(wav, sr, 16000)
        flags = mark_sil(vad, wav)
        wav = wav.unsqueeze(0)

        encoding = encode(hubert, wav, layer)
        for w in range(max(wav_df["word_id"])):
            word_df = wav_df[wav_df["word_id"] == w]

            clean_encoding = cut_and_clean_encoding(
                encoding.squeeze(0),
                flags,
                [word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]],
            )
            if clean_encoding != []:
                codes, _ = segment(
                    clean_encoding.numpy(), kmeans.cluster_centers_, gamma
                )

            save_path = (
                save_dir
                / str(gamma)
                / path.relative_to(wav_dir)
                / f"{path.stem}_{w}.npy"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(save_path, codes)


def main(gamma, layer, audio_dir, align_path, save_dir, audio_ext):
    align_df = pd.read_csv(align_path)

    paths, sample_size = sample_files(
        audio_dir=audio_dir, audio_ext=audio_ext, sample_size=-1
    )

    print(f"Sample size: {sample_size}")
    get_units(paths, align_df, audio_dir, gamma, layer, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process audio files and extract units."
    )
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "audio_dir", type=Path, help="Path to directory containing audio files."
    )
    parser.add_argument("align_path", type=Path, help="Path to alignment CSV file.")
    parser.add_argument(
        "save_dir", type=Path, help="Directory to save extracted features."
    )
    parser.add_argument(
        "--audio_ext",
        type=str,
        default=".flac",
        help="Audio file extension (default: .flac)",
    )

    args = parser.parse_args()
    main(
        args.gamma,
        args.layer,
        args.audio_dir,
        args.align_path,
        args.save_dir,
        args.audio_ext,
    )

# python encode.py 0.1 7 data/dev-clean data/alignments/dev-clean/alignments.csv features/
