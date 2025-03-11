from pathlib import Path
from tqdm import tqdm
import numpy as np
import igraph as ig
import leidenalg as la


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


def build_graph_from_temp(temp_dir, total_chunks):
    g = ig.Graph()

    total_size = get_total_size(temp_dir, total_chunks)
    sample_size = get_n(total_size)
    print(f"total_size: {total_size}, sample_size: {sample_size}")

    g.add_vertices(sample_size)

    for i in tqdm(range(400), desc="Getting Temp Info"):
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

        # Add edges if they exist
        if edges:
            g.add_edges(edges)
            g.es[-len(weights) :]["weight"] = (
                weights  # Assign weights only to newly added edges
            )

    return g


def get_n(length_list):
    return int(1 + np.sqrt(1 + 8 * length_list)) // 2


def build_graph(rows, cols, vals, sample_size):
    g = ig.Graph()
    g.add_vertices(sample_size)

    # Create a boolean mask for values < 0.4
    mask = vals < 0.4

    # Directly create edges as tuples (node1, node2, weight)
    edges_with_weights = np.column_stack((rows[mask], cols[mask], vals[mask]))

    g.add_edges(edges_with_weights[:, :2].tolist())  # Only (row, col) pairs
    g.es["weight"] = edges_with_weights[:, 2]  # Assign weights separately

    return g


def main():
    gamma = 0.1
    temp_dir = Path(f"output/{gamma}/temp")

    g = build_graph_from_temp(temp_dir, 400)
    g.write_pickle(f"output/{gamma}/graph.pkl")

    partition = la.find_partition(g, la.ModularityVertexPartition)
    print(partition)


if __name__ == "__main__":
    main()
