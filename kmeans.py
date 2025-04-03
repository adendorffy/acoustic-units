import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import argparse


def chunk_paths(paths, chunk_size):
    for i in range(0, len(paths), chunk_size):
        yield paths[i : i + chunk_size]


def main(
    audio_dir: Path,
    model_name: str,
    layer: int,
    n_clusters: int,
    batch_size: int = 500,
):
    raw_features_dir = Path("raw_features") / audio_dir / model_name
    feature_paths = list(raw_features_dir.rglob("**/*.npy"))
    print(f"Found {len(feature_paths)} feature files in {raw_features_dir}")
    if len(feature_paths) == 0:
        raise ValueError(f"No feature files found in {raw_features_dir}")

    model_out = (
        Path("kmeans_models") / f"kmeans_{model_name}_layer{layer}_k{n_clusters}.pkl"
    )
    if model_out.exists():
        print(f"KMeans model already exists at {model_out}. Exiting.", flush=True)
        return
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=0, batch_size=batch_size
    )

    for chunk in tqdm(
        chunk_paths(feature_paths, chunk_size=batch_size),
        desc="Training MiniBatchKMeans",
        total=len(feature_paths) // batch_size,
        unit="chunk",
    ):
        encodings = []

        for path in chunk:
            try:
                features = np.load(path)
                if features.shape[0] > 1000:
                    features = features[::3]  # optional subsampling
                encodings.append(features)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")

        if encodings:
            X = np.concatenate(encodings, axis=0)
            kmeans.partial_fit(X)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, model_out)
    print(f"KMeans model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        type=Path,
        default="librispeech/train-clean-100",
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
    main(args.audio_dir, args.model, args.layer, args.n_clusters)
