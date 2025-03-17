from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter
import itertools
import statistics
import editdistance
import argparse
import re


def transcribe_clusters(df, phones, texts):
    """
    Convert a DataFrame with node-cluster mappings to a list of (cluster_id, text),
    ensuring the node index exists in texts.
    """
    tuple_phones = []
    for id, word_phones in enumerate(phones):
        word_phones_tuple = tuple(word_phones[0].split(" "))
        word_phones_tuple = tuple(
            re.sub(r"[012]", "", phn)
            for phn in word_phones_tuple
            if phn != "sil" and phn != "sp"
        )
        text = texts[id]
        tuple_phones.append((id, word_phones_tuple, text))

    cluster_tuples = []
    seen_nodes = set()  # To track nodes we've already added

    for node_id, cluster in tqdm(
        zip(df["node"], df["cluster"]),
        total=len(df["node"]),
        desc="Creating Clusters",
    ):
        for node, phone, word in tuple_phones:
            if node_id == node and node_id not in seen_nodes:
                cluster_tuples.append((cluster, phone, word))
                seen_nodes.add(node_id)  # Mark this node as added
                break  # Exit loop early once node is matched

    return cluster_tuples


def print_clusters(dist_per_cluster):
    cluster_counters = {}

    for cluster_id, group_list, dist in dist_per_cluster:
        words_phones = [("-".join(phn), wrd) for _, phn, wrd in group_list]
        cluster_counters[cluster_id] = Counter(words_phones)  # Count per cluster

    for cluster_id, counter in cluster_counters.items():
        print(
            f"{'-' * 50}\nCluster {cluster_id}: {dist_per_cluster[cluster_id][2]}\n{'-' * 50}"
        )
        for (phoneme, word), count in sorted(counter.items(), key=lambda x: -x[1]):
            print(f"{phoneme:8} [{word:5}] -> {count} times")


def get_phones_and_texts(gamma, align_dir):
    cache_path = Path("features/texts_and_phones.csv")

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
        word_phones = word_df["phones"].iloc[0].split(",")
        word_phones = " ".join(word_phones)
        phones.append(word_phones)

    df = pd.DataFrame({"text": texts, "phones": phones})
    df.to_csv(cache_path, index=False)
    print(f"Saved texts to {cache_path}")

    return df["phones"].apply(lambda x: tuple(x.split(","))), df["text"].tolist()


def distance(p, q):
    """Compute normalized edit distance between two strings."""
    length = max(len(p), len(q))
    if length <= 0:
        return 0.0
    return editdistance.eval(p, q) / length


def ned(clusters, num_clusters):
    """Compute the normalized edit distance (NED) within each cluster."""
    if not clusters:
        return 0

    clusters = sorted(clusters, key=lambda x: x[0])

    distances = []
    distances_per_cluster = []
    for idx, group in tqdm(
        itertools.groupby(clusters, key=lambda x: x[0]),
        total=num_clusters,
        desc="Clustering",
    ):
        group_list = list(group)

        if len(group_list) < 2:
            continue
        clust_distances = []
        for p, q in itertools.combinations(group_list, 2):
            d = distance(p[1], q[1])
            distances.append(d)
            clust_distances.append(d)
        distances_per_cluster.append(
            (idx, group_list, statistics.mean(clust_distances))
        )

    return statistics.mean(distances), distances_per_cluster


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
        phone_clusters = transcribe_clusters(best_partition_df, phones, texts)
        ned_val, dist_p_cluster = ned(phone_clusters, num_clusters - diff)
        print(f"NED: {ned_val}")
        print_clusters(dist_p_cluster)

        update_readme(gamma, best_res, ned_val, diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a given partition.")
    parser.add_argument("gamma", type=float, help="Gamma value for processing.")
    parser.add_argument(
        "align_dir", type=Path, help="Alignment directory for getting true texts."
    )
    args = parser.parse_args()

    main(args.gamma, args.align_dir)
