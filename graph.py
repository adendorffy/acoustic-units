import argparse
import pickle
from pathlib import Path
import numpy as np
import igraph as ig


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
    print(f"💎 Total pairwise entries: {total_size}, Sample size: {sample_size}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph from distance chunks.")
    parser.add_argument("model", type=str, help="Gamma value for processing.")
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument(
        "out_dir", type=Path, help="Directory for output and distances."
    )
    parser.add_argument(
        "threshold", type=float, default=0.4, help="Distance threshold."
    )
    args = parser.parse_args()

    dist_dir = (
        args.out_dir
        / args.model
        / f"layer{args.layer}"
        / f"gamma{args.gamma}"
        / "distances"
    )
    output_dir = args.out_dir / args.model / f"layer{args.layer}" / f"gamma{args.gamma}"
    output_dir.mkdir(exist_ok=True, parents=True)

    graph_path = output_dir / f"graph_t{args.threshold}.pkl"
    if not graph_path.exists():
        graph = build_graph_from_chunks(dist_dir, args.threshold)
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"Graph saved to {graph_path}")
    else:
        print(f"Graph already exists at {graph_path}")
