import argparse
import pickle
from pathlib import Path

import leidenalg as la
import igraph as ig
import pandas as pd

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
    graph_path = output_dir / f"graph_t{threshold}.pkl"

    if not graph_path.exists():
        print(f"{graph_path} does not exist. Please run the graph building step first.")
        return

    with open(graph_path, "rb") as f:
        g = pickle.load(f)
    print(f"Loaded graph from {graph_path}")

    partition_pattern = output_dir.glob("partition_r*.csv")
    partition_files = list(partition_pattern)

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
        ouput_path = output_dir / f"t{threshold}_partition_r{best_res:.6f}.csv"
        best_partition_df.to_csv(ouput_path, index=False)

        print(f"Partition saved to {ouput_path}", flush=True)
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
    parser = argparse.ArgumentParser(description="Run clustering on precomputed graph.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("out_dir", type=Path, help="Output directory base path.")
    parser.add_argument("--num_clusters", default=DEV_CLEAN_CLUSTERS, type=int)
    parser.add_argument("--threshold", default=0.4, type=float)
    parser.add_argument("--resolution", default=0.02, type=float)
    args = parser.parse_args()

    cluster(
        args.gamma,
        args.layer,
        args.out_dir,
        threshold=args.threshold,
        num_clusters=args.num_clusters,
        initial_res=args.resolution,
    )
