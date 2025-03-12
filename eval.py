from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import itertools
import statistics
import editdistance
import argparse


def transcribe_clusters(df, texts):
    """
    Convert a DataFrame with node-cluster mappings to a list of (cluster_id, text).

    Parameters:
    - df: Pandas DataFrame with "node" and "cluster" columns
    - texts: List of text corresponding to each node

    Returns:
    - List of (cluster_id, text) tuples
    """
    cluster_transcriptions = list(
        zip(df["cluster"], df["node"].map(lambda x: texts[x]))
    )

    return cluster_transcriptions


def print_clusters(cluster_transcriptions):
    # Dictionary to store all text per cluster
    cluster_texts = defaultdict(list)

    # Group all text by cluster_id
    for cluster_id, txt in cluster_transcriptions:
        cluster_texts[cluster_id].append(txt)

    # Print all texts in each cluster
    for cluster_id, texts in cluster_texts.items():
        if len(texts) > 1:
            print(f"Cluster {cluster_id}: {' | '.join(texts)}\n")


def get_texts(gamma, align_dir):
    cache_path = Path(f"features/{gamma}/texts.csv")

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        texts = df["text"].tolist()
        print(f"Loaded texts from {cache_path}")
        return texts

    paths = sorted(
        Path(f"features/{gamma}").rglob("**/*.npy"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    align_df = pd.read_csv(align_dir / "alignments.csv")

    texts = []
    for path in tqdm(paths, desc="Appending Text"):
        filename_parts = path.stem.split("_")
        wav_df = align_df[align_df["filename"] == filename_parts[0]]
        word_df = wav_df[wav_df["word_id"] == int(filename_parts[1])]
        texts.append(str(word_df["text"].iloc[0]))

    df = pd.DataFrame({"text": texts})
    df.to_csv(cache_path, index=False)
    print(f"Saved texts to {cache_path}")

    return texts


def distance(p, q):
    """Compute normalized edit distance between two strings."""
    length = max(len(p), len(q))
    return (
        editdistance.eval(p, q) / length if length > 0 else 1
    )  # Avoid division by zero


def ned(discovered):
    """Compute the normalized edit distance (NED) within each cluster."""
    if not discovered:
        return 0

    discovered = sorted(discovered, key=lambda x: x[0])

    distances = []
    for _, group in itertools.groupby(discovered, key=lambda x: x[0]):
        group_list = list(group)

        if len(group_list) < 2:
            continue

        for p, q in itertools.combinations(group_list, 2):
            p_1 = str(p[1])
            q_1 = str(q[1])
            d = distance(p_1, q_1)
            distances.append(d)

    return statistics.mean(distances) if distances else 0


def update_readme(gamma, best_res, ned_value, readme_path="README.md"):
    """
    Updates the README.md file to include the latest gamma, best resolution, and NED values.
    """
    new_entry = f"| {gamma:.2f} | {best_res:.4f} | {ned_value:.3f} |\n"

    # Read the current README content
    with open(readme_path, "r") as f:
        lines = f.readlines()

    # Find where to insert the new row (after the header row)
    for i, line in enumerate(lines):
        if "| Gamma | Best Resolution" in line:  # Find the table header
            insert_idx = i + 2  # Skip the header and separator
            break
    else:
        insert_idx = len(lines)  # If no table exists, append at the end

    # Insert the new entry
    lines.insert(insert_idx, new_entry)

    # Write back to the README
    with open(readme_path, "w") as f:
        f.writelines(lines)

    print(f"Updated README.md with gamma={gamma}, res={best_res}, NED={ned_value}")


def main(gamma, res, alignment_dir):
    partition = pd.read_csv(f"output/{gamma}/best_partition_r{round(res, 3)}.csv")
    texts = get_texts(gamma, alignment_dir)

    cluster_transcriptions = transcribe_clusters(partition, texts)
    ned_val = ned(cluster_transcriptions)
    print(f"NED: {ned_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a given partition.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument("res", type=float, help="Resolution value for processing.")
    parser.add_argument(
        "align_dir", type=Path, help="Alignment directory for getting true texts."
    )
    args = parser.parse_args()

    main(args.gamma, args.res, args.align_dir)
