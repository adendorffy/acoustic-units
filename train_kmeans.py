import torch
import torchaudio
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
from tqdm import tqdm
import argparse


def main(model_name: str, layer: int):
    sr = 16000
    audio_dir = Path("librispeech/audio")
    audio_ext = ".flac"

    audio_paths = list(audio_dir.rglob(f"**/*{audio_ext}"))
    print(f"Encoding {len(audio_paths)} audio files")

    # Load model based on command-line arg
    try:
        bundle = getattr(torchaudio.pipelines, model_name)
    except AttributeError:
        raise ValueError(f"Invalid model name: {model_name}")

    model = bundle.get_model()
    model.eval()

    all_features = []
    for path in tqdm(audio_paths, desc="Encoding speech"):
        waveform, sr_loaded = torchaudio.load(path)
        if sr_loaded != sr:
            waveform = torchaudio.functional.resample(waveform, sr_loaded, sr)

        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer)
            encoding = features[-1].squeeze(0)  # [T, D]

        all_features.append(encoding)

    X = torch.cat(all_features, dim=0).numpy()

    n_clusters = 100
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    model_out = f"models/kmeans_{model_name.lower()}_layer{layer}_k{n_clusters}.pkl"
    joblib.dump(kmeans, model_out)
    print(f"KMeans model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )
    parser.add_argument(
        "layer", type=int, default=9, help="Layer number to extract features from"
    )

    args = parser.parse_args()
    main(args.model, args.layer)
