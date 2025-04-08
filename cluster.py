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


def write_to_list(ind_df, node_to_cluster, partition_dir, partiton_file):
    print("Converting output to .list format...", flush=True)

    class_to_fragments = defaultdict(list)

    for filename, group_df in ind_df.groupby("filename"):
        output_file = partition_dir / f"{filename}.list"
        word_start = 0.0
        with open(output_file, "w") as f:
            for _, row in group_df.iterrows():
                word_end = row["end"]
                global_node_id = row.name

                cluster_id = node_to_cluster.get(global_node_id, -1)
                f.write(f"{word_end:.2f} {cluster_id}\n")

                class_to_fragments[cluster_id].append((filename, word_start, word_end))
                word_start = word_end
    print(f".list format output saved in {partition_dir}", flush=True)

    with open(partiton_file, "w") as f:
        for classnb in sorted(class_to_fragments.keys()):
            f.write(f"Class {classnb}\n")
            for filename, onset, offset in class_to_fragments[classnb]:
                f.write(f"{filename} {onset:.2f} {offset:.2f}\n")
            f.write("\n")

    print(f"✅ Combined output written to: {output_file}")


def cluster(
    model: str,
    layer: int,
    gamma: float,
    features_dir: Path,
    threshold: float = 0.4,
    initial_res: float = 0.02,
    num_clusters: int = DEV_CLEAN_CLUSTERS,
):
    output_dir = Path("graphs") / model / f"layer{layer}" / f"gamma{gamma}"
    graph_path = output_dir / f"graph_t{threshold}.pkl"
    ind_df = pd.read_csv(features_dir / "paths.csv")

    if not graph_path.exists():
        print(f"{graph_path} does not exist. Please run the graph building step first.")
        return

    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded graph from {graph_path}")

    partition_dir = output_dir / f"partition_t{threshold}"
    partiton_file = partition_dir / f"{model}_l{layer}_g{gamma}_t{threshold}.txt"
    if partition_dir.exists() and partiton_file.exists():
        print(
            f"Partition already exists in {partition_dir} with output file {partiton_file}. Skipping clustering."
        )

        # node_to_cluster = dict(
        #     zip(range(len(best_partition.membership)), best_partition.membership)
        # )

        # write_to_list(ind_df, node_to_cluster, partition_dir, partiton_file)
        return

    partition_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting clustering with initial resolution: {initial_res:.6f}", flush=True)
    _, best_partition = adaptive_res_search(
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

    node_to_cluster = dict(
        zip(range(len(best_partition.membership)), best_partition.membership)
    )

    write_to_list(ind_df, node_to_cluster, partition_dir, partiton_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster graph using Leiden algorithm."
    )
    parser.add_argument("model", type=str, help="Model name for processing.")
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")

    parser.add_argument(
        "features_dir", type=Path, help="Directory for features and paths."
    )
    parser.add_argument(
        "threshold", type=float, default=0.4, help="Distance threshold."
    )
    parser.add_argument("resolution", default=0.02, type=float)
    args = parser.parse_args()

    cluster(
        args.model,
        args.layer,
        args.gamma,
        args.features_dir,
        args.threshold,
        initial_res=args.resolution,
        num_clusters=DEV_CLEAN_CLUSTERS,
    )
