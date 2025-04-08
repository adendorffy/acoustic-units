import argparse
import pickle
from pathlib import Path
import numpy as np
import igraph as ig
import pandas as pd
import re
from typing import Iterable


def get_total_size(dist_dir: Path) -> int:
    total_chunks = sum(1 for _ in dist_dir.rglob("vals_*.npy"))
    print(f"Get total_size for {dist_dir} [total_chunks: {total_chunks}]")
    total_size = sum(
        np.load(dist_dir / f"vals_{i}.npy").shape[0] for i in range(total_chunks)
    )
    return total_size, total_chunks


def get_n(length_list: int) -> int:
    return int((1 + np.sqrt(1 + 8 * length_list)) // 2)


def clean_phones(phones: Iterable[str]) -> list[str]:
    unwanted = {"sil", "sp", "spn", "", "nan"}
    cleaned = []

    if isinstance(phones, float):
        phones = [str(phones)]
    else:
        phones = phones.split(",")

    for phone in phones:
        phone = re.sub(r"\d+", "", phone).strip()
        if phone not in unwanted:
            cleaned.append(phone)

    return cleaned


def build_graph_from_chunks(
    dist_dir: Path,
    path_df: pd.DataFrame,
    align_df: pd.DataFrame,
    threshold: float = 0.4,
) -> ig.Graph:
    total_size, total_chunks = get_total_size(dist_dir)
    sample_size = get_n(total_size)
    print(f"💎 Total pairwise entries: {total_size}, Sample size: {sample_size}")

    valid_keys = [
        f"{row.filename}_{row.word_id}"
        for row in align_df.itertuples(index=False)
        if clean_phones(row.phones)  # not empty
    ]
    valid_df = path_df[path_df["filename"].isin(valid_keys)].copy()
    valid_df["vertex_id"] = (
        valid_df["filename"].str.strip().str.replace("'", "", regex=False)
    )

    id_to_label = dict(zip(valid_df.index, valid_df["vertex_id"]))

    g = ig.Graph()
    g.add_vertices(list(valid_df["vertex_id"]))
    prev_progress = -1

    for i in range(total_chunks):
        rows = np.load(dist_dir / f"rows_{i}.npy")
        cols = np.load(dist_dir / f"cols_{i}.npy")
        vals = np.load(dist_dir / f"vals_{i}.npy")

        threshold_mask = vals < threshold
        r_filtered = rows[threshold_mask]
        c_filtered = cols[threshold_mask]
        w_filtered = vals[threshold_mask]

        for r, c, w in zip(r_filtered, c_filtered, w_filtered):
            if r in id_to_label and c in id_to_label:
                g.add_edge(id_to_label[r], id_to_label[c], weight=max(w, 1e-10))

        progress = int((i / total_chunks) * 100)
        if progress % 5 == 0 and progress > prev_progress:
            print(f"🟢 Progress: {progress}% ({i}/{total_chunks} chunks)", flush=True)
            prev_progress = progress

    g.vs["filename"] = list(valid_df["filename"])
    g.vs["index"] = list(valid_df.index)

    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph from distance chunks.")
    parser.add_argument(
        "model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )

    parser.add_argument(
        "layer", type=int, default=8, help="Layer number to extract features from"
    )
    parser.add_argument("gamma", type=float, default=1.0, help="Gamma")
    parser.add_argument(
        "n_clusters", type=int, default=100, help="Number of clusters for KMeans"
    )

    parser.add_argument(
        "align_dir",
        type=Path,
        help="Path to the directory containing alignments.",
    )
    parser.add_argument(
        "threshold", type=float, default=0.4, help="Distance threshold."
    )
    args = parser.parse_args()

    dist_dir = (
        Path("distances")
        / args.model
        / f"layer{args.layer}"
        / f"gamma{args.gamma}"
        / f"k{args.n_clusters}"
    )
    path_df = pd.read_csv(
        f"features/{args.model}/layer{args.layer}/gamma{args.gamma}/k{args.n_clusters}/paths.csv",
    )
    align_df = pd.read_csv(args.align_dir / "alignments.csv")
    output_dir = (
        Path("graphs") / args.model / f"layer{args.layer}" / f"gamma{args.gamma}"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    graph_path = output_dir / f"graph_t{args.threshold}.pkl"

    if not graph_path.exists():
        graph = build_graph_from_chunks(dist_dir, path_df, align_df, args.threshold)
        with open(graph_path, "wb") as f:
            pickle.dump(graph, f)
        print(f"Graph saved to {graph_path}")
    else:
        print(f"Graph already exists at {graph_path}")
