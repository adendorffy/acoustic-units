import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import joblib
from tqdm import tqdm
import argparse


def main(raw_features_dir: Path, model_name: str, layer: int, n_clusters: int):
    model_name = model_name.upper()

    feature_paths = list(raw_features_dir.rglob("**/*.npy"))
    print(f"Loading {len(feature_paths)} feature files from {raw_features_dir}")

    kmeans = KMeans(n_clusters=n_clusters)
    for chunk in tqdm(
        feature_paths,
        desc="Processing audio files",
    ):
        encodings = []
        for path in chunk:
            features = np.load(path)
            encodings.append(features)

        kmeans.fit(encodings)

    model_out = f"models/kmeans_{model_name}_layer{layer}_k{n_clusters}.pkl"

    joblib.dump(kmeans, model_out)
    print(f"KMeans model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        type=str,
        default="librispeech/audio",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "model",
        type=str,
        default="wavlm_base",
        help="Model name from torchaudio.pipelines (in lowercase)",
    )
    parser.add_argument(
        "layer", type=int, default=7, help="Layer number to extract features from"
    )
    parser.add_argument(
        "n_clusters",
        type=int,
        default=100,
        help="Number of clusters for KMeans (100, 200, 500, 1000)",
    )

    args = parser.parse_args()
    main(args.audio_dir, args.audio_ext, args.model, args.layer, args.n_clusters)
