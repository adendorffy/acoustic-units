import itertools
import editdistance
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from tqdm import tqdm


def ned(word_clusters, print_pure=False, print_inpure=True):
    distances = []
    for i, clust in tqdm(
        enumerate(word_clusters), total=len(word_clusters), desc="Calculating NED"
    ):
        if len(clust) > 1:
            clust_dist = []

            for p, q in itertools.combinations(clust, 2):
                dist = editdistance.eval(p.true_word, q.true_word)
                clust_dist.append(dist)
                distances.append(dist)

            if any(dist > 0 for dist in clust_dist) and print_inpure or print_pure:
                print(f"Cluster {i}: {statistics.mean(clust_dist)}")
                words = [j.true_word for j in clust]
                print(", ".join(words))
                print()

    return statistics.mean(distances) if distances else 0


def pairwise_edit_dist_mat(dist_mat, title, true_words):
    condensed_dist_mat = squareform(dist_mat)
    dist_df_hub = pd.DataFrame(dist_mat, index=true_words, columns=true_words)
    linked = linkage(condensed_dist_mat, method="average")
    order = leaves_list(linked)
    reordered_dist_df = dist_df_hub.iloc[order, order]

    plt.Figure(figsize=(8, 6))
    sns.heatmap(reordered_dist_df, cmap="viridis")
    plt.title(title)
    plt.show()


def words_from_word_units(word_clusters):
    clusters = []
    for cluster in word_clusters:
        words = []
        for word in cluster:
            words.append(word.true_word)
        clusters.append(words)
    return clusters


def clusters_purity(just_words_clusters):
    count = 0
    total = len(just_words_clusters)
    visited = set(just_words_clusters[0])
    for c in range(1, total):
        clust_set = set(just_words_clusters[c])

        if visited.intersection(clust_set):
            count += 1

        visited = visited.union(clust_set)

    return count / total, total


def calculate_duplicate_clusters(clusters, print_clusters=False):
    normalized_clusters = [Counter([j.true_word for j in clust]) for clust in clusters]

    cluster_counts = Counter(map(lambda d: frozenset(d.items()), normalized_clusters))
    total_duplicates = sum(count for count in cluster_counts.values() if count > 1)

    if print_clusters:
        print(
            f"Total duplicate clusters (considering word frequency): {sum(count for count in cluster_counts.values() if count > 1)}"
        )
        print("Duplicate clusters and their counts:")
        for cluster, count in cluster_counts.items():
            if count > 1:
                print(f"{dict(cluster)}: {count} times")
    return cluster_counts, total_duplicates


def dendogram(dist_mat, true_words):
    condensed_dist_mat = squareform(dist_mat)
    linked = linkage(condensed_dist_mat, method="average")

    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=true_words, leaf_font_size=6)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample")
    plt.ylabel("Distance")
    plt.show()
