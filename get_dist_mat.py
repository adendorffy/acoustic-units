from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import editdistance
from line_profiler import profile
from typing import Generator, List, Tuple, Dict


def pair_generator(num_paths: int) -> Generator[Tuple[int, int], None, None]:
    """
    Generator function that yields all unique index pairs (i, j)
    where i < j. This ensures each pair is processed only once.
    """
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            yield i, j


def get_batch_of_paths(
    num_paths: int, chunk_limit: int = 100
) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Creates and yields chunks of index pairs for processing.

    Args:
        num_paths (int): Number of paths (features) to process.
        chunk_limit (int): Maximum number of pairs in each chunk.

    Yields:
        List of tuples containing index pairs (i, j).
    """
    pairs = pair_generator(num_paths)
    chunk: List[Tuple[int, int]] = []

    for idx, (i, j) in enumerate(pairs, 1):
        chunk.append((i, j))

        if idx % chunk_limit == 0:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


@profile
def load_units(path: Path) -> np.ndarray:
    """
    Loads a NumPy array from the given file path.

    Args:
        path (Path): Path to the .npy file.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    return np.load(path)


@profile
def calculate_distance_per_chunk(
    chunk_pair: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[int, int, float]:
    """
    Calculates the normalized edit distance for a given pair of feature sequences.

    Args:
        chunk_pair (dict): Dictionary with a single key-value pair where:
            - Key: Tuple (i, j) representing the indices of the feature pair.
            - Value: Tuple (feature_i, feature_j) containing the feature sequences.

    Returns:
        tuple: (index_i, index_j, normalized edit distance).
    """

    id_1, id_2 = tuple(chunk_pair.keys())[0]
    feature_1, feature_2 = tuple(chunk_pair.values())[0]

    length = np.max([len(feature_1), len(feature_2)])

    dist = 0
    if length > 0:
        dist = editdistance.eval(feature_1, feature_2) / length

    return (id_1, id_2, dist)


def info_to_csv(csv_path: str, file_map: Dict[int, Path]) -> None:
    """
    Saves the mapping of indices to filenames in a CSV file.

    Args:
        csv_path (str): Path to the output CSV file.
        file_map (dict): Dictionary mapping index (int) to filename (Path).
    """

    rows: List[Tuple[int, Path]] = [(file, file_map[int(file)]) for file in file_map]
    df = pd.DataFrame(rows, columns=["id", "filename"])
    df.to_csv(csv_path, index=False)


@profile
def main() -> None:
    """
    Main function that:
    - Loads feature data from files.
    - Computes pairwise edit distances in chunks.
    - Saves distance matrix and file metadata.
    """
    feat_dir = Path("features/dusted_units/0.2/")

    file_map = {}
    features = []
    for i, feature in enumerate(feat_dir.rglob("**/*.npy")):
        file_map[i] = feature.stem
        features.append(load_units(feature))

    sample_size = i
    print(f"sample_size: {sample_size}")
    dist_mat = np.zeros((sample_size, sample_size), dtype=np.float32)

    csv_path = "output/dusted/info.csv"
    dist_mat_out_path = Path("output/dusted/dist_mat.npz")
    dist_mat_out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_limit = 1000000
    num_pairs = sample_size * (sample_size - 1) // 2
    num_chunks = (num_pairs + chunk_limit - 1) // chunk_limit

    for chunk in tqdm(
        get_batch_of_paths(sample_size, chunk_limit=chunk_limit),
        total=num_chunks,
        desc="Processing chunks",
    ):
        chunk_units = [{(i, j): (features[i], features[j])} for i, j in chunk]

        with Pool(7) as pool:
            chunk_results = pool.map(calculate_distance_per_chunk, chunk_units)

        for i, j, dist in chunk_results:
            dist_mat[i, j] = dist

    info_to_csv(csv_path, file_map)
    np.savez_compressed(dist_mat_out_path, dist_mat)


if __name__ == "__main__":
    main()
