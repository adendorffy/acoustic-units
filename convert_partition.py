from pathlib import Path
import pandas as pd
import argparse


def get_partition_path(gamma: float, layer: int, threshold: float, output_dir: Path):
    base_dir = output_dir / str(gamma) / str(layer)

    print(f"🔍 Searching for partition in: {base_dir}")

    matches = list(base_dir.glob(f"t{threshold}_partition_r*.csv"))

    if not matches:
        print("❌ No partition file found. First calculate partition.")
        return None

    if len(matches) > 1:
        print("⚠️ Multiple partition files found. Using the first match.")

    res = str(matches[0]).split("_r")[1]
    res = res.split(".")[1]
    resolution = float(f"0.{res}")
    return matches[0], resolution


def convert_to_list(gamma: float, layer: int, threshold: float, output_dir: Path):
    partition_path, resolution = get_partition_path(gamma, layer, threshold, output_dir)
    partition_df = pd.read_csv(partition_path)

    out_dir = output_dir / str(gamma) / str(layer) / str(resolution)
    out_dir.mkdir(parents=True, exist_ok=True)

    align_df = pd.read_csv(output_dir / "alignments_aligned_to_features.csv")
    node_to_cluster = dict(zip(partition_df["node"], partition_df["cluster"]))

    print("Converting output to .list format...", flush=True)
    for filename, group_df in align_df.groupby("filename"):
        output_file = out_dir / f"{filename}.list"
        with open(output_file, "w") as f:
            for _, row in group_df.iterrows():
                word_end = row["end"]
                global_node_id = row.name

                cluster_id = node_to_cluster.get(global_node_id, -1)
                f.write(f"{word_end:.2f} {cluster_id}\n")
    print(f".list format output saved in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process feature distances and save results in chunks."
    )
    parser.add_argument(
        "gamma", type=float, help="Gamma value used for feature extraction."
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("threshold", type=float, help="Threshold at whic to extract.")

    parser.add_argument(
        "output_dir", type=Path, help="Path to the directory where output is stored."
    )

    args = parser.parse_args()

    convert_to_list(args.gamma, args.layer, args.threshold, args.output_dir)
