from pathlib import Path
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from eval import get_sim_mat
import numpy as np
from scipy.sparse import csr_matrix
import heapq
from tqdm import tqdm


def basic_graph_clustering(similarity_matrix, threshold=0.5):
    """
    Performs basic clustering using connected components on a sparse similarity matrix.

    Parameters:
    - similarity_matrix (scipy.sparse matrix): Sparse similarity matrix.
    - threshold (float): Minimum similarity value to consider an edge.

    Returns:
    - dict: Node-to-cluster mapping.
    """
    # Ensure matrix is in CSR format
    similarity_matrix = sp.csr_matrix(similarity_matrix)

    # Create a graph where edges exist for similarity >= threshold
    sources, targets = similarity_matrix.nonzero()
    weights = similarity_matrix.data

    # Filter edges based on threshold
    edges = [(s, t) for s, t, w in zip(sources, targets, weights) if w >= threshold]

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find connected components (each component is a cluster)
    components = list(nx.connected_components(G))

    # Assign cluster labels
    node_to_cluster = {
        node: i for i, cluster in enumerate(components) for node in cluster
    }

    return node_to_cluster  # Dictionary {node: cluster}


class SparseHierarchicalClustering:
    def __init__(self, k=2):
        self.k = k  # Number of clusters

    def fit(self, sparse_dist_matrix):
        """Performs Agglomerative Hierarchical Clustering on a sparse distance matrix"""
        n = sparse_dist_matrix.shape[0]  # Number of data points
        clusters = {i: [i] for i in range(n)}  # Each point starts as its own cluster
        priority_queue = self.build_priority_queue(sparse_dist_matrix)

        with tqdm(total=n - self.k, desc="Clustering progress") as pbar:
            while len(clusters) > self.k:
                # Get closest cluster pair (min distance)
                while priority_queue:
                    dist, c1, c2 = heapq.heappop(priority_queue)
                    if c1 in clusters and c2 in clusters:
                        break

                # Merge clusters c2 into c1
                clusters[c1].extend(clusters[c2])
                del clusters[c2]

                # Update distance matrix efficiently
                self.update_distance_queue(
                    priority_queue, sparse_dist_matrix, clusters, c1, c2
                )
                pbar.update(1)

        self.labels_ = self.assign_labels(clusters, n)
        return self

    def build_priority_queue(self, sparse_dist_matrix):
        """Create a min-heap priority queue for cluster distances"""
        priority_queue = []
        coo = (
            sparse_dist_matrix.tocoo()
        )  # Convert sparse matrix to COO format for iteration

        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j:  # Avoid duplicate pairs
                heapq.heappush(priority_queue, (v, i, j))

        return priority_queue

    def update_distance_queue(
        self, priority_queue, sparse_dist_matrix, clusters, c1, c2
    ):
        """Update priority queue after merging clusters (single-linkage)"""
        for other in list(clusters.keys()):
            if other != c1:
                min_dist = min(
                    sparse_dist_matrix[p1, p2]
                    if sparse_dist_matrix[p1, p2] > 0
                    else np.inf
                    for p1 in clusters[c1]
                    for p2 in clusters[other]
                )
                heapq.heappush(priority_queue, (min_dist, c1, other))

    def assign_labels(self, clusters, n):
        """Assign cluster labels to original data points"""
        labels = np.zeros(n, dtype=int)
        for cluster_id, points in enumerate(clusters.values()):
            for p in points:
                labels[p] = cluster_id
        return labels


def main():
    align_path = Path("data/alignments/dev-clean/alignments.csv")
    csv_path = Path("output/0.2/info.csv")

    align_df = pd.read_csv(align_path)
    info_df = pd.read_csv(csv_path)

    sparse_dist_mat = sp.load_npz("output/0.2/sparse_dist_mat.npz")

    sim_mat = get_sim_mat(sparse_dist_mat)

    # labels = basic_graph_clustering(sim_mat, 0.9)

    # print(f"Num clusrers: {len(labels)}")

    k = 13967
    hc = SparseHierarchicalClustering(k=k)
    hc.fit(sparse_dist_mat.tocsr())
    print("Cluster labels:", hc.labels_)


if __name__ == "__main__":
    main()
