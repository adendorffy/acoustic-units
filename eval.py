from pathlib import Path
from tqdm import tqdm
import numpy as np
import igraph as ig
import leidenalg as la
import pandas as pd
from collections import defaultdict
import itertools
import statistics
import editdistance
import argparse


def distance(p, q):
    """Compute normalized edit distance between two strings."""
    length = max(len(p), len(q))
    return (
        editdistance.eval(p, q) / length if length > 0 else 1
    )  # Avoid division by zero


def ned(discovered):
    """Compute the normalized edit distance (NED) within each cluster."""
    if not discovered:
        return 0

    discovered = sorted(discovered, key=lambda x: x[0])

    distances = []
    for _, group in itertools.groupby(discovered, key=lambda x: x[0]):
        group_list = list(group)

        if len(group_list) < 2:
            continue

        for p, q in itertools.combinations(group_list, 2):
            d = distance(p[1], q[1])
            distances.append(d)

    return (
        statistics.mean(distances) if distances else 0
    )  # Ensure no division by empty list


def get_total_size(temp_dir, total_chunks):
    total_size = sum(
        np.load(temp_dir / f"temp_rows_{i}.npy").shape[0]
        for i in tqdm(range(total_chunks), desc="Calculating total")
    )
    return total_size


def concat_temp_files(temp_dir, save_dir, total_chunks):
    rows, cols, vals = [], [], []
    elements = ["rows", "cols", "vals"]

    first_chunk = np.load(temp_dir / "temp_rows_0.npy")
    dtype = first_chunk.dtype

    total_size = get_total_size(temp_dir, total_chunks)

    for element in elements:
        file_path = save_dir / f"{element}.npy"
        merged_data = np.memmap(file_path, dtype=dtype, mode="w+", shape=(total_size,))

        index = 0
        for i in tqdm(range(total_chunks), desc=f"Writing {element} to disk"):
            temp_data = np.load(temp_dir / f"temp_{element}_{i}.npy")
            merged_data[index : index + temp_data.shape[0]] = temp_data

            index += temp_data.shape[0]

        merged_data.flush()
        del merged_data

    return rows, cols, vals


def get_texts(gamma, align_dir):
    paths = (p for p in Path(f"features/{gamma}").rglob("**/*.npy"))
    align_df = pd.read_csv(align_dir / "alignments.csv")
    sorted_paths = sorted(paths, key=lambda x: int(x.stem.split("_")[-1]))
    texts = []
    for path in tqdm(sorted_paths, desc="Appending Text"):
        filename_parts = path.stem.split("_")
        wav_df = align_df[align_df["filename"] == filename_parts[0]]
        word_df = wav_df[wav_df["word_id"] == int(filename_parts[1])]
        print(path)
        texts.append(str(word_df["text"].iloc[0]))

    return texts


def build_graph_from_temp(temp_dir, total_chunks):
    total_size = get_total_size(temp_dir, total_chunks)
    sample_size = get_n(total_size)
    print(f"total_size: {total_size}, sample_size: {sample_size}")

    g = ig.Graph()
    g.add_vertices(sample_size)

    for i in tqdm(range(total_chunks), desc="Getting Temp Info"):
        temp_rows = np.load(temp_dir / f"temp_rows_{i}.npy")
        temp_cols = np.load(temp_dir / f"temp_cols_{i}.npy")
        temp_vals = np.load(temp_dir / f"temp_vals_{i}.npy")

        mask = temp_vals < 0.4
        filtered_rows = temp_rows[mask]
        filtered_cols = temp_cols[mask]
        filtered_vals = temp_vals[mask]

        # Convert edges and weights to lists
        edges = list(zip(map(int, filtered_rows), map(int, filtered_cols)))
        weights = list(map(float, filtered_vals))

        weights = [w if w > 0 else 1e-10 for w in weights]

        # Add edges if they exist
        if edges:
            g.add_edges(edges)

            # Assign weights only to newly added edges
            if weights:
                g.es[-len(weights) :].set_attribute_values("weight", weights)

    return g


def get_n(length_list):
    return int(1 + np.sqrt(1 + 8 * length_list)) // 2


def transcribe_clusters(partition, texts):
    clusters = {i: [] for i in set(partition.membership)}

    # Assign nodes to clusters
    for node, cluster_id in enumerate(partition.membership):
        clusters[cluster_id].append(node)  # Append node index to the respective cluster

    cluster_transcriptions = []
    for cluster_id, words in clusters.items():
        for w in words:
            cluster_transcriptions.append((cluster_id, texts[w]))

    return cluster_transcriptions


def print_clusters(cluster_transcriptions):
    # Dictionary to store all text per cluster
    cluster_texts = defaultdict(list)

    # Group all text by cluster_id
    for cluster_id, txt in cluster_transcriptions:
        cluster_texts[cluster_id].append(txt)

    # Print all texts in each cluster
    for cluster_id, texts in cluster_texts.items():
        if len(texts) > 1:
            print(f"Cluster {cluster_id}: {' | '.join(texts)}\n")


def main(gamma, num_clusters, res, align_dir):
    temp_dir = Path(f"output/{gamma}/temp")
    texts = get_texts(gamma, align_dir)

    g = build_graph_from_temp(temp_dir, 400)
    g.write_pickle(f"output/{gamma}/graph.pkl")

    partition = la.find_partition(
        g, la.CPMVertexPartition, weights="weight", resolution_parameter=res
    )
    actual_clusters = len(set(partition.membership))

    cluster_transcriptions = transcribe_clusters(partition, texts)
    # print_clusters(cluster_transcriptions)

    print(f"Cluster difference: {abs(actual_clusters - num_clusters)}")
    print(f"NED: {ned(cluster_transcriptions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run graph-based clustering on text data."
    )
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument("num_clusters", type=int, help="Target number of clusters.")
    parser.add_argument(
        "res", type=float, help="Resolution parameter for Leiden clustering."
    )
    parser.add_argument("align_dir", type=Path, help="Path to alignment directory.")

    args = parser.parse_args()
    main(args.gamma, args.num_clusters, args.res, args.align_dir)

# python eval.py 0.1 13967 0.0277 data/alignments/dev-clean/
