import editdistance
import itertools
import statistics
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import seaborn as sns
import pandas as pd
from scipy.sparse import issparse


def get_true_words(info_df: pd.DataFrame, align_df: pd.DataFrame):
    """
    Efficiently extracts corresponding words from `align_df` based on `info_df` filenames.

    Args:
        info_df (pd.DataFrame): DataFrame containing `filename` column with "file_wordID".
        align_df (pd.DataFrame): DataFrame containing `filename`, `word_id`, and `text`.

    Returns:
        List[str]: List of corresponding words or "_" if no match is found.
    """
    if "word_id" not in info_df:
        split_cols = info_df["filename"].str.split("_", expand=True)
        info_df[["filename", "word_id"]] = split_cols
        info_df["word_id"] = info_df["word_id"].astype(int)
    merged_df = info_df.merge(align_df, on=["filename", "word_id"], how="left")

    merged_df["text"] = merged_df["text"].fillna("_")
    return merged_df["text"].tolist()


def convert_to_word_clusters(int_clusters, text_arr):
    word_clusters = []
    for cluster in int_clusters:
        words = []
        for i in cluster:
            words.append(text_arr[i])
        word_clusters.append(words)
    return word_clusters


def ned(word_clusters):
    distances = []

    for cluster in tqdm(word_clusters, desc="Calculating NED"):
        for p, q in itertools.combinations(cluster, 2):
            dist = editdistance.eval(p, q)
            distances.append(dist)

    mean_dist = 0
    if distances:
        mean_dist = statistics.mean(distances)

    return mean_dist


def pairwise_edit_dist_mat(dist_mat, title, true_words):
    """
    Visualizes a pairwise edit distance matrix, adapted for sparse representations.

    Parameters:
    - dist_mat (scipy.sparse or np.ndarray): The pairwise distance matrix (sparse or dense).
    - title (str): Title for the heatmap.
    - true_words (list): List of words corresponding to the matrix rows/columns.

    Returns:
    - None (displays heatmap)
    """

    # Convert sparse matrix to dense if necessary
    if issparse(dist_mat):
        dist_mat = dist_mat.toarray()  # Convert to dense NumPy array

    # Convert to condensed form for clustering
    dist_mat += dist_mat.T
    condensed_dist_mat = squareform(dist_mat)  # Ensure proper format

    # Create a DataFrame for visualization
    dist_df_hub = pd.DataFrame(dist_mat, index=true_words, columns=true_words)

    # Perform hierarchical clustering
    linked = linkage(condensed_dist_mat, method="average")
    order = leaves_list(linked)

    # Reorder distance matrix based on clustering
    reordered_dist_df = dist_df_hub.iloc[order, order]

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(reordered_dist_df, cmap="viridis", xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.show()


def print_clusters(word_clusters, print_pure=False, print_inpure=True):
    for i, clust in enumerate(word_clusters):
        if len(clust) > 1:
            clust_dist = []

            for p, q in itertools.combinations(clust, 2):
                dist = editdistance.eval(p, q)
                clust_dist.append(dist)

            if any(dist > 0 for dist in clust_dist) and print_inpure or print_pure:
                print(f"Cluster {i}: {statistics.mean(clust_dist)}")
                words = [j for j in clust]
                print(", ".join(words))
                print()


def get_sim_mat(dist_mat):
    dist_mat = dist_mat.tocsr()

    # Convert distances to similarities for nonzero elements
    max_dist = dist_mat.data.max()
    similarity_matrix = dist_mat.copy()
    similarity_matrix.data = max_dist - similarity_matrix.data

    return similarity_matrix
