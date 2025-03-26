import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import leidenalg as la
import igraph as ig
import pandas as pd

DEV_CLEAN_CLUSTERS: int = 13_967


def get_total_size(dist_dir: Path) -> int:
    total_chunks = sum(1 for _ in dist_dir.rglob("vals_*.npy"))
    print(f"Get total_size for {dist_dir} [total_chunks: {total_chunks}]")
    total_size = sum(
        np.load(dist_dir / f"vals_{i}.npy").shape[0] for i in range(total_chunks)
    )
    return total_size, total_chunks


def get_n(length_list: int) -> int:
    return int((1 + np.sqrt(1 + 8 * length_list)) // 2)


def build_graph_from_chunks(dist_dir: Path, threshold: float = 0.4) -> ig.Graph:
    total_size, total_chunks = get_total_size(dist_dir)
    sample_size = get_n(total_size)
    print(f"🧮 Total pairwise entries: {total_size}, Sample size: {sample_size}")

    g = ig.Graph()
    g.add_vertices(sample_size)
    prev_progress = -1

    for i in range(total_chunks):
        rows = np.load(dist_dir / f"rows_{i}.npy")
        cols = np.load(dist_dir / f"cols_{i}.npy")
        vals = np.load(dist_dir / f"vals_{i}.npy")

        mask = vals < threshold
        edges = list(zip(rows[mask], cols[mask]))
        weights = vals[mask].astype(float)

        weights = np.where(weights > 0, weights, 1e-10).tolist()

        if edges:
            g.add_edges(edges)
            g.es[-len(weights) :].set_attribute_values("weight", weights)

        progress = int((i / total_chunks) * 100)
        if progress % 5 == 0 and progress > prev_progress:
            print(f"🟢 Progress: {progress}% ({i}/{total_chunks} chunks)", flush=True)
            prev_progress = progress

    return g


def adaptive_res_search(
    g: ig.Graph,
    num_clusters: int,
    initial_res: float = 0.02,
    alpha: float = 0.001,
    max_iters: int = 50,
    tol: float = 1e-6,
    patience: int = 3,
    min_alpha: float = 1e-5,
    alpha_boost: float = 1.1,
) -> Tuple[float]:
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

        if diff == 0:
            break

        grad = 1 if actual_clusters < num_clusters else -1

        if prev_diff is not None and diff >= prev_diff:
            worsening_steps += 1
        else:
            worsening_steps = 0

        prev_diff = diff

        if worsening_steps >= patience:
            print(
                f"Reversing direction & boosting step size (α → {alpha * alpha_boost:.6f}).",
                flush=True,
            )
            grad *= -1
            alpha *= alpha_boost
            worsening_steps = 0

        new_res = max(0.001, min(res + alpha * grad, 5.0))

        if abs(new_res - res) < tol:
            print("Resolution is stabilizing. Stopping search.")
            break

        res = new_res
        alpha = max(alpha * 0.95, min_alpha)

    return best_res, best_partition


def cluster(
    gamma: float,
    layer: int,
    out_dir: Path,
    threshold: float = 0.4,
    initial_res: float = 0.02,
    num_clusters: int = DEV_CLEAN_CLUSTERS,
):
    output_dir = out_dir / str(gamma) / str(layer)
    dist_dir = out_dir / "distances" / str(gamma) / str(layer)

    if not dist_dir.exists():
        print(f"{dist_dir} does not exist. First calculate distances.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    graph_path = output_dir / f"graph_t{threshold}.pkl"
    partition_pattern = output_dir.glob("partition_r*.csv")
    partition_files = list(partition_pattern)

    if graph_path.exists():
        with open(graph_path, "rb") as f:
            g = pickle.load(f)
        print(f"Loaded precomputed graph from {graph_path}")
    else:
        g = build_graph_from_chunks(dist_dir, threshold)
        with open(graph_path, "wb") as f:
            pickle.dump(g, f)
        print(f"Graph built and saved to {graph_path}", flush=True)

    if not partition_files:
        best_res, best_partition = adaptive_res_search(
            g, num_clusters, initial_res=initial_res
        )
        best_partition_df = pd.DataFrame(
            {
                "node": range(len(best_partition.membership)),
                "cluster": best_partition.membership,
            }
        )
        ouput_path = output_dir / f"partition_r{best_res:.6f}.csv"
        best_partition_df.to_csv(ouput_path, index=False)

        print(
            f"Partition saved to {str(output_dir) + f'/partition_r{best_res:.6f}.csv'}",
            flush=True,
        )
        return best_res, best_partition_df
    else:
        res_partitions = [
            (float(p.stem.split("_r")[1]), pd.read_csv(p)) for p in partition_files
        ]
        if not res_partitions:
            print("No partitions found.")
            return
        best_res = res_partitions[0][0]
        best_partition_df = res_partitions[0][1]
        return best_res, best_partition_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run graph-based clustering on text data."
    )
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "out_dir", type=Path, help="Path to the directory to store output."
    )
    parser.add_argument(
        "--num_clusters",
        default=DEV_CLEAN_CLUSTERS,
        type=int,
        help="Target number of clusters.",
    )
    parser.add_argument(
        "--threshold",
        default=0.4,
        type=float,
        help="Distance threshold for graph edges.",
    )

    args = parser.parse_args()
    cluster(
        args.gamma,
        args.layer,
        args.out_dir,
        args.threshold,
        num_clusters=args.num_clusters,
    )
