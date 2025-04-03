import torch
import torchaudio
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import random


def main(audio_dir: Path, audio_ext: str, model_name: str, layer: int):
    features_dir = Path("raw_features") / audio_dir / model_name
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = list(features_dir.rglob("**/*.npy"))
    model_name = model_name.upper()
    sr = 16000

    audio_paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    audio_paths = random.sample(audio_paths, min(2703, len(audio_paths)))
    print(f"Encoding {len(audio_paths)} audio files from {audio_dir}", flush=True)

    try:
        bundle = getattr(torchaudio.pipelines, model_name.upper())
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name.upper()}")

    model = bundle.get_model()
    model.eval()

    for path in tqdm(
        audio_paths,
        desc="Processing audio files",
        total=len(audio_paths),
    ):
        save_path = (
            features_dir / path.relative_to(audio_dir).parent / f"{path.stem}.npy"
        )
        if save_path in feature_paths:
            continue

        waveform, sr = torchaudio.load(path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer)
            encoding = features[layer - 1].squeeze(0).cpu().numpy()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, encoding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir", type=Path, help="Path to the directory containing audio files"
    )
    parser.add_argument(
        "audio_ext",
        type=str,
        help="Audio file extension (e.g., .wav, .flac)",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name (e.g., WAVLM_BASE)",
    )
    parser.add_argument(
        "layer",
        type=int,
        help="Layer number to extract features from",
    )

    args = parser.parse_args()
    main(args.audio_dir, args.audio_ext, args.model_name, args.layer)
