import argparse
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import editdistance


def pair_generator(
    num_paths: int, start: int = 0
) -> Generator[Tuple[int, int], None, None]:
    for i in range(start, num_paths):
        for j in range(i + 1, num_paths):
            yield i, j


def get_batch_of_paths(
    num_paths: int, chunk_limit: int = 5_000_000
) -> Generator[List[Tuple[int, int]], None, None]:
    chunk: List[Tuple[int, int]] = []
    for idx, pair in enumerate(pair_generator(num_paths), 1):
        chunk.append(pair)
        if idx % chunk_limit == 0:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def calculate_edit_distance(
    pair: Tuple[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, int, float]:
    (idx_1, idx_2), (feature_1, feature_2) = pair
    length = max(len(feature_1), len(feature_2))

    return (
        idx_1,
        idx_2,
        editdistance.eval(feature_1, feature_2) / length if length else 1.0,
    )


def calculate_dist_files(
    model: str,
    layer: int,
    gamma: float,
    feat_dir: Path,
    output_dir: Path,
    chunk_limit: int = 5_000_000,
):
    feature_dir = feat_dir / model / f"layer{layer}" / f"gamma{gamma}"
    paths = list(feature_dir.rglob("**/*.npy"))
    sample_size = len(paths)
    if sample_size < 2:
        print(
            f"Not enough samples in {str(feature_dir)} to compute pairwise distances.",
            flush=True,
        )
        return
    print(f"Loading {sample_size} Features.. from {feature_dir}", flush=True)
    features = [np.load(path) for path in paths]

    num_pairs = sample_size * (sample_size - 1) // 2
    num_batches = (num_pairs + chunk_limit - 1) // chunk_limit
    prev_progress = -1

    print(
        f"Pairs: {num_pairs}, Batches: {num_batches}",
        flush=True,
    )
    out_dir = output_dir / model / f"layer{layer}" / f"gamma{gamma}" / "distances"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = list(out_dir.rglob("vals_*.npy"))
    if len(out_paths) == num_batches:
        print(f"All batches already processed and saved in {out_dir}.", flush=True)
        return
    print("Calculating pairwise distances...", flush=True)

    chunk_idx = 0
    for batch in get_batch_of_paths(sample_size, chunk_limit):
        rows, cols, vals = zip(
            *[
                calculate_edit_distance(((i, j), (features[i], features[j])))
                for i, j in batch
            ]
        )
        np.save(
            out_dir / f"rows_{chunk_idx}.npy",
            np.array(rows, dtype=np.int32),
        )
        np.save(
            out_dir / f"cols_{chunk_idx}.npy",
            np.array(cols, dtype=np.int32),
        )
        np.save(
            out_dir / f"vals_{chunk_idx}.npy",
            np.array(vals, dtype=np.float32),
        )

        progress = int((chunk_idx / num_batches) * 100)
        if progress % 5 == 0 and progress > prev_progress:
            print(
                f"🟢 Progress: {progress}% ({chunk_idx}/{num_batches} files)",
                flush=True,
            )
            prev_progress = progress
        chunk_idx += 1

    print(f"All batches processed. Results saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        required=True,
        help="Model name (e.g., wavlm, wav2vec2)",
    )

    parser.add_argument(
        "layer",
        type=int,
        required=True,
        help="Layer number for feature extraction",
    )
    parser.add_argument(
        "gamma",
        type=float,
        required=True,
        help="Gamma value for distance calculation",
    )
    args = parser.parse_args()

    calculate_dist_files(
        args.model,
        args.layer,
        args.gamma,
        Path("features"),
        Path("output"),
    )
