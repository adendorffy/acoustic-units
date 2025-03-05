from tqdm import tqdm
import numpy as np
import scipy.sparse as sp


class UnionFind:
    """Efficient Union-Find (Disjoint Set) with path compression and union by rank."""

    def __init__(self, size: int):
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=int)

    def find(self, node: int) -> int:
        """Find the root with path compression."""
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1: int, node2: int) -> None:
        """Union by rank optimization."""
        root1, root2 = self.find(node1), self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def cluster_sparse(dist_mat: sp.spmatrix, dist_threshold: float):
    """
    Clusters points using a sparse distance matrix and Union-Find.

    Args:
        dist_mat (sp.spmatrix): Upper triangular sparse distance matrix.
        dist_threshold (float): Maximum distance for clustering.

    Returns:
        List[List[int]]: List of clusters.
    """
    num_nodes = dist_mat.shape[0]
    uf = UnionFind(num_nodes)

    # Extract nonzero elements (upper triangular values)
    i_indices, j_indices, values = sp.find(
        dist_mat
    )  # Efficiently retrieves stored distances

    # Filter based on threshold
    mask = values < dist_threshold
    i_indices, j_indices = i_indices[mask], j_indices[mask]

    # Perform union operations in batches
    for i, j in tqdm(
        zip(i_indices, j_indices), total=len(i_indices), desc="Clustering"
    ):
        uf.union(i, j)

    # Group nodes by root parent
    clusters = {}
    for node in tqdm(range(num_nodes), desc="Grouping Clusters"):
        root = uf.find(node)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(node)

    return list(clusters.values())


def to_sparse_upper_chunked(
    dist_mat: np.ndarray, chunk_size: int = 5000, save_path: str = "sparse_dist_mat.npz"
):
    """
    Convert a large full distance matrix to a sparse upper triangular format in chunks.

    Args:
        dist_mat (np.ndarray): Large full distance matrix (NxN).
        chunk_size (int): Number of rows to process at a time.
        save_path (str): Path to save the sparse matrix.

    Returns:
        sp.spmatrix: Sparse upper triangular matrix (COO format).
    """
    num_nodes = dist_mat.shape[0]
    row_indices = []
    col_indices = []
    values = []

    # Process the upper triangle in chunks
    for start in tqdm(
        range(0, num_nodes, chunk_size), desc="Converting to Sparse", unit="chunk"
    ):
        end = min(start + chunk_size, num_nodes)  # Define chunk range
        i_indices, j_indices = np.triu_indices(
            end - start, k=1
        )  # Get upper triangle indices

        i_indices += start  # Offset row indices
        row_indices.extend(i_indices)
        col_indices.extend(j_indices)
        values.extend(dist_mat[i_indices, j_indices])  # Append only relevant values

    # Convert to sparse COO format
    sparse_matrix = sp.coo_matrix(
        (values, (row_indices, col_indices)), shape=(num_nodes, num_nodes)
    )

    # Save compressed sparse matrix
    sp.save_npz(save_path, sparse_matrix)

    return sparse_matrix


# Example usage:
# sparse_dist_mat = to_sparse_upper_chunked(large_dist_mat, chunk_size=5000, save_path="sparse_dist_mat.npz")


def graph_cluster(dist_mat, dist_threshold):
    num_nodes = dist_mat.shape[0]
    graph = {i: set() for i in range(num_nodes)}

    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if dist_mat[i, j] < dist_threshold:
                graph[i].add(j)
                graph[j].add(i)

    clusters = []
    visited = set()

    def bfs(start_node):
        """Traverse a cluster using BFS"""
        queue = [start_node]
        cluster = []

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(graph[node])

        return cluster

    for node in tqdm(range(num_nodes), desc="Clustering"):
        if node not in visited:
            new_cluster = bfs(node)
            clusters.append(new_cluster)

    return clusters
