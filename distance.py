from typing import Generator, List, Tuple, Dict
import numpy as np
from pathlib import Path
import editdistance
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool


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


def calculate_distance_per_chunk_pair(
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


def process_chunks(
    feat_dir: Path, info_csv_path: Path, dist_mat_out_path: Path, chunk_limit: int
) -> None:
    dist_mat_out_path.parent.mkdir(parents=True, exist_ok=True)
    """
    Processes feature files from a directory, computes pairwise distances between them in chunks, 
    and saves the resulting distance matrix.

    Args:
        feat_dir (Path): Directory containing `.npy` feature files.
        info_csv_path (Path): Path to save the CSV file mapping indices to file names.
        dist_mat_out_path (Path): Path to save the compressed distance matrix.
        chunk_limit (int): Maximum number of pairs to process per chunk.

    Returns:
        None
    """

    file_map = {}
    features = []
    for i, feature in enumerate(feat_dir.rglob("**/*.npy")):
        file_map[i] = feature.stem
        features.append(np.load(feature))

    sample_size = len(features)
    dist_mat = np.zeros((sample_size, sample_size))

    num_pairs = sample_size * (sample_size - 1) // 2
    num_chunks = (num_pairs + chunk_limit - 1) // chunk_limit

    for chunk in tqdm(
        get_batch_of_paths(sample_size, chunk_limit=chunk_limit),
        total=num_chunks,
        desc="Processing Chunks",
    ):
        chunk_units = [{(i, j): (features[i], features[j])} for i, j in chunk]

        with Pool(7) as pool:
            chunk_results = pool.map(calculate_distance_per_chunk_pair, chunk_units)

        for i, j, dist in chunk_results:
            dist_mat[i, j] = dist

    info_to_csv(info_csv_path, file_map)
    np.savez_compressed(dist_mat_out_path, dist_mat)
