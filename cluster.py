from pathlib import Path
from tqdm import tqdm
import numpy as np
import igraph as ig
import leidenalg as la
import pandas as pd
import argparse
import pickle


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


def build_graph_from_temp(temp_dir, total_chunks, threshold=0.4):
    total_size = get_total_size(temp_dir, total_chunks)
    sample_size = get_n(total_size)
    print(f"total_size: {total_size}, sample_size: {sample_size}")

    g = ig.Graph()
    g.add_vertices(sample_size)

    for i in tqdm(range(total_chunks), desc="Getting Temp Info"):
        temp_rows = np.load(temp_dir / f"temp_rows_{i}.npy")
        temp_cols = np.load(temp_dir / f"temp_cols_{i}.npy")
        temp_vals = np.load(temp_dir / f"temp_vals_{i}.npy")

        mask = temp_vals < threshold
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


def adaptive_res_search(
    g,
    num_clusters,
    initial_res=0.02,
    alpha=0.001,
    max_iters=50,
    tol=1e-6,
    patience=3,  # Allow worsening steps before reversing
    min_alpha=1e-5,  # Prevent alpha from shrinking too much
    alpha_boost=1.1,  # Increase alpha slightly when reversing
):
    """
    Find the best resolution parameter adaptively.

    Parameters:
    - g: Graph for clustering
    - num_clusters: Target number of clusters
    - initial_res: Starting resolution parameter
    - alpha: Step size (learning rate), decays over time
    - max_iters: Max number of iterations
    - tol: Tolerance for detecting stabilization
    - patience: Number of bad steps before reversing
    - min_alpha: Minimum step size to prevent getting stuck
    - alpha_boost: Multiplier to increase step size after reversing

    Returns:
    - best_res: The best found resolution parameter
    - best_partition: The corresponding partition
    """

    res = initial_res
    min_diff = float("inf")
    best_res = res
    best_partition = None
    prev_diff = None  # Track previous diff
    prev_res = None

    worsening_steps = 0  # Count worsening steps

    for t in range(1, max_iters + 1):
        partition = la.find_partition(
            g,
            la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=42,  # Set seed for reproducibility
        )
        actual_clusters = len(set(partition.membership))
        diff = abs(actual_clusters - num_clusters)

        # Track the best resolution so far
        if diff < min_diff:
            min_diff = diff
            best_res = res
            best_partition = partition
            worsening_steps = 0  # Reset worsening step counter

        print(f"Iteration {t}: res={res:.6f}, Cluster difference={diff}")

        if min_diff == 0:  # Stop early if an exact match is found
            break

        # Compute adaptive gradient
        if prev_diff is not None:
            if diff < prev_diff:
                grad = -1  # Moving in the right direction
                worsening_steps = 0  # Reset worsening count
            else:
                grad = 1  # Moving in the wrong direction
                worsening_steps += 1  # Track consecutive bad steps
        else:
            grad = 1 if actual_clusters < num_clusters else -1  # Initial direction

        prev_diff = diff  # Update previous difference

        # If we hit the patience threshold, reverse direction & boost step size
        if worsening_steps >= patience:
            print(
                f"Too many worsening steps ({worsening_steps}), reversing direction and boosting step size."
            )
            grad *= -1
            alpha *= alpha_boost  # Increase step size slightly
            worsening_steps = 0  # Reset patience counter

        prev_res = res
        res += alpha * grad  # Apply gradient update

        # If `res` stabilizes (small changes), stop
        if prev_res is not None and abs(prev_res - res) < tol:
            print("Res is stabilizing. Abort.")
            break

        # Prevent resolution from going out of bounds
        res = max(0.001, min(res, 5.0))  # Keep within reasonable range

        # Adaptive learning rate decay (but keep it above `min_alpha`)
        alpha *= 0.95  # Reduce step size gradually
        alpha = max(alpha, min_alpha)  # Ensure alpha does not become too small

    return best_res, best_partition


def main(gamma, num_clusters=13967, build_graph=False, compute_res=False):
    temp_dir = Path(f"output/{gamma}/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    graph_path = Path(f"output/{gamma}/graph.pkl")

    if not build_graph and graph_path.exists():
        with open(graph_path, "rb") as f:
            g = pickle.load(f)
        print(f"Loaded precomputed graph from {graph_path}")
    else:
        g = build_graph_from_temp(temp_dir, 399)
        g.write_pickle(str(graph_path))
        print(f"Graph built and saved to {graph_path}")

    partition_pattern = Path(f"output/{gamma}").glob("partition_r*.csv")
    partition_files = list(partition_pattern)

    if not partition_files or compute_res:
        # No existing partitions found, run the search
        best_res, best_partition = adaptive_res_search(g, num_clusters)

        # Convert best_partition to a DataFrame
        best_partition_df = pd.DataFrame(
            {
                "node": range(len(best_partition.membership)),  # Node IDs
                "cluster": best_partition.membership,  # Cluster assignments
            }
        )

        # Save to CSV
        best_partition_df.to_csv(
            f"output/{gamma}/partition_r{round(best_res, 3)}.csv", index=False
        )
    else:
        # Load existing partitions
        res_partitions = [
            (float(p.stem.split("_r")[1]), pd.read_csv(p)) for p in partition_files
        ]

        # Find the partition with the minimum resolution
        best_res, best_partition_df = min(res_partitions, key=lambda x: x[0])

    # Ensure best_partition_df is used for further processing
    actual_clusters = len(set(best_partition_df["cluster"]))
    diff = abs(actual_clusters - num_clusters)

    print(f"Best resolution found: {best_res:.3f} with cluster difference: {diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run graph-based clustering on text data."
    )
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument(
        "--num_clusters", default=13967, type=int, help="Target number of clusters."
    )

    parser.add_argument(
        "--build_graph", action="store_true", help="Don't use preloaded graph."
    )
    parser.add_argument(
        "--compute_res", action="store_true", help="Compute resolution."
    )

    args = parser.parse_args()
    main(args.gamma, args.num_clusters, args.build_graph, args.compute_res)
