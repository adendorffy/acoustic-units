from pathlib import Path
from distance import get_batch_of_paths
from tqdm import tqdm
import scipy.sparse as sp
import editdistance
import numpy as np


def get_features_and_filenames(sorted_paths):
    filenames = []
    features = []
    for path in tqdm(sorted_paths, desc="Appending Features"):
        feature = np.load(path)

        filenames.append(path.stem)
        features.append(feature)

    return features, filenames


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
    ((id_1, id_2), (feature_1, feature_2)) = pair

    max_length = max(len(feature_1), len(feature_2))
    min_length = min(len(feature_1), len(feature_2))

    if min_length == 0:
        return id_1, id_2, 1.0  # Max distance when one feature is empty

    if max_length > 0:
        return id_1, id_2, editdistance.eval(feature_1, feature_2) / max_length

    return id_1, id_2, 0


def main():
    # Process chunks
    gamma = 0.1
    paths = (p for p in Path(f"features/{gamma}").rglob("*.npy"))

    sorted_paths = sorted(paths, key=lambda x: int(x.stem.split("_")[-1]))
    sample_size = len(sorted_paths)

    features, filenames = get_features_and_filenames(sorted_paths)

    rows, cols, vals = [], [], []

    num_pairs = sample_size * (sample_size - 1) // 2
    chunk_limit = 5000000
    num_batches = (num_pairs + chunk_limit - 1) // chunk_limit

    print(f"num_samples: {sample_size}")
    print(f"num_pairs: {num_pairs}")

    chunk_idx = 0
    for batch in tqdm(
        get_batch_of_paths(sample_size, chunk_limit),
        total=num_batches,
        unit="batch",
        mininterval=10.0,
        desc="Processing Batches",
    ):
        for i, j in batch:
            i, j, dist = cal_dist_per_pair(((i, j), (features[i], features[j])))
            rows.append(i)
            cols.append(j)
            vals.append(dist)

        np.save(f"output/{gamma}/temp_rows_{chunk_idx}.npy", rows)
        np.save(f"output/{gamma}/temp_cols_{chunk_idx}.npy", cols)
        np.save(f"output/{gamma}/temp_vals_{chunk_idx}.npy", vals)

        rows, cols, vals = [], [], []
        chunk_idx += 1  # Increment chunk index

    # all_rows, all_cols, all_vals = [], [], []
    # for i in range(chunk_idx):
    #     all_rows.append(np.load(f"output/{gamma}/temp_rows_{i}.npy"))
    #     all_cols.append(np.load(f"output/{gamma}/temp_cols_{i}.npy"))
    #     all_vals.append(np.load(f"output/{gamma}/temp_vals_{i}.npy"))

    # # Merge into single NumPy arrays
    # rows = np.concatenate(all_rows)
    # cols = np.concatenate(all_cols)
    # vals = np.concatenate(all_vals)

    # # Convert to a compressed sparse format for efficient storage
    # print("Saving sparse matrix...")
    # if rows and cols and vals:
    #     dist_sparse = sp.coo_matrix(
    #         (vals, (rows, cols)), shape=(sample_size, sample_size)
    #     )
    #     sp.save_npz(f"output/{gamma}/sparse_dist_mat.npz", dist_sparse)


if __name__ == "__main__":
    main()
