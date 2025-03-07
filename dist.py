from pathlib import Path
from distance import pair_generator, get_batch_of_paths
from tqdm import tqdm
import scipy.sparse as sp
import editdistance
import numpy as np
import pandas as pd
from line_profiler import profile


@profile
def cal_dist_per_pair(pair):
    """
    Calculates the normalized edit distance for a given pair of feature sequences.

    Args:
        chunk_pair (dict): Dictionary with a single key-value pair where:
            - Key: Tuple (i, j) representing the indices of the feature pair.
            - Value: Tuple (feature_i, feature_j) containing the feature sequences.

    Returns:
        tuple: (index_i, index_j, normalized edit distance).
    """
    (id_1, id_2), (feature_1, feature_2) = pair

    max_length = max(len(feature_1), len(feature_2))
    min_length = min(len(feature_1), len(feature_2))

    if min_length == 0:
        return id_1, id_2, 1.0  # Max distance when one feature is empty

    dist = (
        editdistance.eval(feature_1, feature_2) / max_length if max_length > 0 else 1.0
    )
    return id_1, id_2, dist


@profile
def get_features_and_filenames(sorted_paths):
    filenames = []
    features = []
    for path in tqdm(sorted_paths, desc="Appending Features"):
        filenames.append(path.stem)
        feature = np.load(path)
        features.append(feature)
    return features, filenames


@profile
def main():
    # Process chunks
    gamma = 0.1
    paths = list(Path(f"features/{gamma}").rglob("*.npy"))

    sorted_paths = sorted(paths, key=lambda x: int(x.stem.split("_")[-1]))
    sample_size = len(sorted_paths)

    features, filenames = get_features_and_filenames(sorted_paths)

    start_i = 0
    rows = []
    cols = []
    vals = []

    print(f"Start i: {start_i}")

    num_pairs = sample_size * (sample_size - 1) // 2
    chunk_limit = 10000
    num_batches = (num_pairs + chunk_limit - 1) // chunk_limit

    print(f"num_samples: {sample_size}")
    print(f"num_pairs: {num_pairs}")

    for batch in tqdm(
        get_batch_of_paths(sample_size, chunk_limit),
        total=num_batches,
        unit="batch",
        mininterval=1.0,
        desc="Processing Batches",
    ):
        for i, j in batch:
            i, j, dist = cal_dist_per_pair(((i, j), (features[i], features[j])))
            rows.append(i)
            cols.append(j)
            vals.append(dist)

    # Convert to a compressed sparse format for efficient storage
    if rows and cols and vals:
        dist_sparse = sp.coo_matrix(
            (vals, (rows, cols)), shape=(sample_size, sample_size)
        )
        sp.save_npz(f"output/{gamma}/sparse_dist_mat.npz", dist_sparse)


if __name__ == "__main__":
    main()
