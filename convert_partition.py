from pathlib import Path
import pandas as pd
import argparse
from collections import defaultdict


def get_partition_path(model: str, layer: int, threshold: float, output_dir: Path):
    base_dir = output_dir / str(model) / str(layer)

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


def convert_to_list(model: str, layer: int, threshold: float, output_dir: Path):
    partition_path, resolution = get_partition_path(model, layer, threshold, output_dir)
    partition_df = pd.read_csv(partition_path)

    out_dir = output_dir / model / str(layer) / str(resolution)
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
    return out_dir


def convert_lists_to_class_format(list_dir: Path, output_file: Path):
    class_to_fragments = defaultdict(list)

    list_files = sorted(list_dir.rglob("*.list"))

    for list_file in list_files:
        filename = list_file.stem
        with open(list_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        onset = 0.0
        for line in lines:
            try:
                offset_str, classnb_str = line.split()
                offset = float(offset_str)
                classnb = int(classnb_str)
                class_to_fragments[classnb].append((filename, onset, offset))
                onset = offset
            except ValueError:
                print(f"⚠️ Malformed line in {list_file}: {line}")

    # Write to output
    with open(output_file, "w") as f:
        for classnb in sorted(class_to_fragments.keys()):
            f.write(f"Class {classnb}\n")
            for filename, onset, offset in class_to_fragments[classnb]:
                f.write(f"{filename} {onset:.2f} {offset:.2f}\n")
            f.write("\n")

    print(f"✅ Combined output written to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process feature distances and save results in chunks."
    )
    # parser.add_argument(
    #     "gamma", type=float, help="Gamma value used for feature extraction."
    # )
    parser.add_argument(
        "model",
        type=str,
        default="HUBERT_BASE",
        help="Model name from torchaudio.pipelines (e.g., HUBERT_BASE, WAV2VEC2_BASE)",
    )
    parser.add_argument("layer", type=int, help="Layer number for processing.")
    parser.add_argument("threshold", type=float, help="Threshold at whic to extract.")

    parser.add_argument(
        "output_dir", type=Path, help="Path to the directory where output is stored."
    )

    args = parser.parse_args()

    list_dir = convert_to_list(args.model, args.layer, args.threshold, args.output_dir)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    convert_lists_to_class_format(
        list_dir,
        output_file=results_dir / f"{args.model}_l{args.layer}_t{args.threshold}.txt",
    )
