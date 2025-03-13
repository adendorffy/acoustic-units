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
    Convert a DataFrame with node-cluster mappings to a list of (cluster_id, text),
    ensuring the node index exists in texts.
    """

    cluster_transcriptions = list(
        zip(
            df["cluster"],
            df["node"].map(lambda x: texts[x]),
        )
    )

    return cluster_transcriptions


def transcribe_clusters_into_phones(df, phones):
    """
    Convert a DataFrame with node-cluster mappings to a list of (cluster_id, text),
    ensuring the node index exists in texts.
    """
    cluster_transcriptions = [
        (cluster, phones[x])
        for cluster, x in zip(df["cluster"], df["node"])
        if phones[x] not in {"sil", "sp"}
    ]

    return cluster_transcriptions


def print_clusters(cluster_transcriptions):
    # Dictionary to store all text per cluster
    cluster_texts = defaultdict(list)

    # Group all text by cluster_id
    for cluster_id, txt in cluster_transcriptions:
        cluster_texts[cluster_id].append(txt)

    # Print all texts in each cluster
    for cluster_id, texts in cluster_texts.items():
        if len(texts) > 10:
            print(
                f"Cluster {cluster_id}: {' | '.join([str(text) for text in texts])}\n"
            )


def get_phones_and_texts(gamma, align_dir):
    cache_path = Path(f"features/{gamma}/texts_and_phones.csv")

    if cache_path.exists():
        df = pd.read_csv(cache_path)
        texts = df["text"].tolist()
        phones = df["phones"].apply(lambda x: tuple(x.split(",")))
        print(f"Loaded texts from {cache_path}")
        return phones, texts

    paths = sorted(
        Path(f"features/{gamma}").rglob("**/*.npy"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    align_df = pd.read_csv(align_dir / "alignments.csv")

    texts = []
    phones = []

    for path in tqdm(paths, desc="Appending Text and Phones"):
        filename_parts = path.stem.split("_")
        wav_df = align_df[align_df["filename"] == filename_parts[0]]
        word_df = wav_df[wav_df["word_id"] == int(filename_parts[1])]
        texts.append(str(word_df["text"].iloc[0]))
        phones.extend(word_df["phones"].apply(lambda x: tuple(x.split(","))).tolist())

    df = pd.DataFrame({"text": texts, "phones": phones})
    df.to_csv(cache_path, index=False)
    print(f"Saved texts to {cache_path}")

    return phones, texts


def distance(p, q):
    """Compute normalized edit distance between two strings."""
    length = max(len(p), len(q))
    return (
        editdistance.eval(p, q) / length if length > 0 else 1
    )  # Avoid division by zero


def ned(clusters):
    """Compute the normalized edit distance (NED) within each cluster."""
    if not clusters:
        return 0

    clusters = sorted(clusters, key=lambda x: x[0])

    distances = []
    for _, group in itertools.groupby(clusters, key=lambda x: x[0]):
        group_list = list(group)

        if len(group_list) < 2:
            continue

        for p, q in itertools.combinations(group_list, 2):
            d = distance(p[1], q[1])
            distances.append(d)

    return statistics.mean(distances)


def update_readme(gamma, best_res, ned_value, diff, readme_path="README.md"):
    """
    Updates the README.md file to include the latest gamma, best resolution, and NED values.
    """
    new_entry = f"| {gamma:.2f} | {best_res:.4f} | {ned_value:.3f} | {diff} |\n"

    # Read the current README content
    with open(readme_path, "r") as f:
        lines = f.readlines()

    if new_entry in lines:
        print("Entry already in README.md. No update needed.")
        return

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


def main(gamma, alignment_dir, num_clusters=13967):
    partition_pattern = Path(f"output/{gamma}").glob("partition_r*.csv")
    partition_files = list(partition_pattern)

    if not partition_files:
        # No existing partitions found, run the search
        print("No partition files found. First calculate partition.")
        return
    else:
        # Load existing partitions
        res_partitions = [
            (float(p.stem.split("_r")[1]), pd.read_csv(p)) for p in partition_files
        ]

        # Find the partition with the minimum resolution
        best_res, best_partition_df = min(res_partitions, key=lambda x: x[0])
        actual_clusters = len(set(best_partition_df["cluster"]))
        diff = abs(actual_clusters - num_clusters)
        phones, texts = get_phones_and_texts(gamma, alignment_dir)
        phone_clusters = transcribe_clusters_into_phones(best_partition_df, phones)
        text_clusters = transcribe_clusters(best_partition_df, texts)

        for id, clust in enumerate(text_clusters):
            if len(clust) > 5:
                print(f"Cluster {id}: {clust}")
        ned_val = ned(phone_clusters)

        print(f"NED: {ned_val}")
        update_readme(gamma, best_res, ned_val, diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a given partition.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument(
        "align_dir", type=Path, help="Alignment directory for getting true texts."
    )
    args = parser.parse_args()

    main(args.gamma, args.align_dir)
