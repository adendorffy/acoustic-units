import argparse
import pickle
from pathlib import Path

import leidenalg as la
import igraph as ig
import pandas as pd
from collections import defaultdict

DEV_CLEAN_CLUSTERS: int = 13_967


def adaptive_res_search(
    g: ig.Graph,
    num_clusters: int,
    initial_res: float = 0.02,
    alpha: float = 0.01,
    max_iters: int = 50,
    tol: float = 1e-6,
    patience: int = 3,
    min_alpha: float = 1e-5,
    alpha_boost: float = 1.1,
    diff_tol: int = 5,
):
    res = initial_res
    best_res, best_partition = res, None
    min_diff = float("inf")
    prev_diff = None
    worsening_steps = 0

    for t in range(1, max_iters + 1):
        partition = la.find_partition(
            g,
            la.CPMVertexPartition,
            weights="weight",
            resolution_parameter=res,
            seed=42,
        )
        actual_clusters = len(set(partition.membership))
        diff = abs(actual_clusters - num_clusters)

        if diff < min_diff:
            min_diff, best_res, best_partition = diff, res, partition
            worsening_steps = 0

        print(
            f"[Iteration {t}] res={res:.6f}, clusters={actual_clusters}, diff={diff}",
            flush=True,
        )

        if min_diff < diff_tol:
            break

        grad = 1 if actual_clusters < num_clusters else -1

        if prev_diff is not None and diff >= prev_diff:
            worsening_steps += 1
        else:
            worsening_steps = 0

        prev_diff = diff

        if worsening_steps >= patience:
            print(
                f"Reversing direction & boosting step size (\u03b1 → {alpha * alpha_boost:.6f}).",
                flush=True,
            )
            grad *= -1
            alpha *= alpha_boost
            worsening_steps = 0

        new_res = max(0.000, min(res + alpha * grad, 5.0))

        if abs(new_res - res) < tol:
            print("Resolution is stabilizing. Stopping search.", flush=True)
            break

        res = new_res
        alpha = max(alpha * 0.95, min_alpha)

    return best_res, best_partition


def write_to_list(ind_df, node_to_cluster, partition_file, align_df):
    class_to_fragments = defaultdict(list)

    # Pre-index for fast lookup
    ind_lookup = dict(zip(ind_df["filename"], ind_df["word_index"]))
    align_df_grouped = align_df.groupby("filename")

    # Fast iteration
    for filename in ind_df["filename"].unique():
        base_filename, word_id = filename.split("_")
        word_id = int(word_id)
        global_node_id = ind_lookup[filename]
        cluster_id = node_to_cluster.get(global_node_id, -1)
        if cluster_id == -1:
            continue

        try:
            file_df = align_df_grouped.get_group(base_filename)
            word_row = file_df[file_df["word_id"] == word_id].iloc[0]
            word_start = float(word_row["word_start"])
            word_end = float(word_row["word_end"])
        except (KeyError, IndexError):
            continue

        class_to_fragments[cluster_id].append((base_filename, word_start, word_end))

    # Write output
    with open(partition_file, "w") as f:
        for classnb in sorted(class_to_fragments):
            f.write(f"Class {classnb}\n")
            for filename, onset, offset in class_to_fragments[classnb]:
                f.write(f"{filename} {onset:.2f} {offset:.2f}\n")
            f.write("\n")

    print(f"✅ Combined output written to: {partition_file}")


def cluster(
    model: str,
    layer: int,
    gamma: float,
    n_clusters: int,
    features_dir: Path,
    align_dir: Path,
    threshold: float = 0.4,
    initial_res: float = 0.02,
    num_clusters: int = DEV_CLEAN_CLUSTERS,
):
    output_dir = (
        Path("partitions")
        / model
        / f"layer{layer}"
        / f"gamma{gamma}"
        / f"k{n_clusters}"
    )
    graph_path = (
        Path("graphs")
        / model
        / f"layer{layer}"
        / f"gamma{gamma}"
        / f"k{n_clusters}"
        / f"graph_t{threshold}.pkl"
    )
    ind_df = pd.read_csv(
        features_dir
        / model
        / f"layer{layer}"
        / f"gamma{gamma}"
        / f"k{n_clusters}"
        / "paths.csv"
    )
    align_df = pd.read_csv(align_dir / "alignments.csv")

    if not graph_path.exists():
        print(f"{graph_path} does not exist. Please run the graph building step first.")
        return

    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded graph from {graph_path}")

    partiton_file = output_dir / f"{model}_l{layer}_g{gamma}_t{threshold}.txt"
    if partiton_file.exists():
        print(f"Partition already exists in {partiton_file}. Skipping clustering.")

        return
    partiton_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting clustering with initial resolution: {initial_res:.6f}", flush=True)
    best_res, best_partition = adaptive_res_search(
        g,
        num_clusters,
        initial_res=initial_res,
        alpha=0.01,
        max_iters=50,
        tol=1e-6,
        patience=3,
        min_alpha=1e-5,
        alpha_boost=1.1,
    )
    if best_res == 0:
        print("Best resolution is 0. Exiting.")
        return
    node_to_cluster = dict(
        zip(range(len(best_partition.membership)), best_partition.membership)
    )

    write_to_list(ind_df, node_to_cluster, partiton_file, align_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster graph using Leiden algorithm."
    )
    parser.add_argument("model", type=str, help="Model name for processing.")
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument("n_clusters", type=int, help="Number of clusters for KMeans.")

    parser.add_argument(
        "features_dir", type=Path, help="Directory for features and paths."
    )
    parser.add_argument("align_dir", type=Path, help="Directory for alignments.")
    parser.add_argument(
        "threshold", type=float, default=0.4, help="Distance threshold."
    )
    parser.add_argument("resolution", default=0.02, type=float)
    args = parser.parse_args()

    cluster(
        args.model,
        args.layer,
        args.gamma,
        args.n_clusters,
        args.features_dir,
        args.align_dir,
        args.threshold,
        initial_res=args.resolution,
        num_clusters=DEV_CLEAN_CLUSTERS,
    )
