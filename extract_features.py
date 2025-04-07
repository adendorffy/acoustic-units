import torch
import torchaudio
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import random


def main(audio_dir: Path, audio_ext: str, model_name: str, layer: int):
    num_sample = 2703
    features_dir = Path("raw_features") / audio_dir / model_name / f"layer{layer}"
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = list(features_dir.rglob("**/*.npy"))
    print(
        f"Found {len(feature_paths)} feature files in {features_dir}",
        flush=True,
    )
    if len(feature_paths) >= num_sample:
        print(
            f"All {num_sample} features already processed and saved in {features_dir}.",
            flush=True,
        )
        return
    model_name = model_name.upper()
    sr = 16000

    audio_paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    audio_paths = random.sample(audio_paths, min(num_sample, len(audio_paths)))
    if len(audio_paths) < num_sample:
        print(
            f"Only {len(audio_paths)} audio files found. "
            f"Please check the audio directory and extension.",
            flush=True,
        )
        return
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

        waveform, sr = torchaudio.load(str(path))
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
    print(
        f"extract_features.py, [{args.audio_dir}, {args.audio_ext}, {args.model_name}, {args.layer}]",
        flush=True,
    )
    main(args.audio_dir, args.audio_ext, args.model_name, args.layer)
