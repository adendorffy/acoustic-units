import numpy as np
import editdistance
from eval import calculate_duplicate_clusters, ned
from tqdm import tqdm
from pathlib import Path
import pandas as pd

def cluster(dist_mat, dist_threshold):
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
        """ Traverse a cluster using BFS """
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

def get_loaded_clusters(words):

    clusters_dict = {}
    for w in words:
        if w.cluster_id in clusters_dict:
            clusters_dict[w.cluster_id].extend([w])
        else:
            clusters_dict[w.cluster_id] = [w]
    
    clusters = []
    for id in clusters_dict:
        clusters.append(clusters_dict[id])

    return clusters

def get_word_clusters(int_clusters, words):

    word_clusters = []
    for i, clust in tqdm(enumerate(int_clusters), total=len(int_clusters), desc="Getting Word Clusters"):
        words_ = []
        for k in range(len(clust)):
            word_list = [w for w in words if w.id == clust[k]]
            word = word_list[0]
            word.add_cluster_id(id=i)
            words_.append(word)
        word_clusters.append(words_)
    
    return word_clusters

def get_cluster_centroids(word_clusters):
    centroids = []

    for i, clust in tqdm(enumerate(word_clusters), total=len(word_clusters), desc="Calculating Cluster Centroids"):
        units_stack = [word.clean_encoding for word in clust]
        max_len = max([len(units) for units in units_stack])
        padded_units_stack = [
            np.pad(units, (0, max_len - len(units)), mode='constant', constant_values=0)
            for units in units_stack
        ]
        centroid_encoding = np.mean(padded_units_stack, axis=0)
        dist = [editdistance.eval(centroid_encoding, clust[j].clean_encoding) for j in range(len(clust))]
        closest_word = clust[np.argmin(dist)]
        centroids.append(closest_word)
    
    return centroids

def get_distance_to_centroids(word_clusters, centroids):
    words = []
    
    for clust in tqdm(word_clusters, desc="Get Distances to Centroids"):
        for word in clust:
            distances_to_centroids = [
                editdistance.eval(word.clean_encoding, centroids[i].clean_encoding) 
                for i in range(len(centroids))
            ]
            word.add_cluster_id(id=np.argmin(distances_to_centroids))
            words.append(word)
    return words

def recalculate_clusters(words):
    clusters = []
    for i in range(max([word.cluster_id for word in words])+1):
        words_in_i = [word for word in words if word.cluster_id == i]
        clusters.append(words_in_i)

    return clusters

def get_best_clusters(word_clusters, current_ned, max_iter=10, tolerance=1e-5):
    _, duplicate_counts = calculate_duplicate_clusters(word_clusters, print_clusters=False)
    
    best_clusters = word_clusters
    best_ned = current_ned
    best_duplicate_count = duplicate_counts
    
    i = 0
    print(f"Iteration {i}: NED: {current_ned:.6f}, Duplicates: {duplicate_counts}")
    while i < max_iter:
        centroids = get_cluster_centroids(word_clusters)
        words = get_distance_to_centroids(word_clusters, centroids)

        word_clusters = recalculate_clusters(words)
        _, duplicate_counts = calculate_duplicate_clusters(word_clusters, print_clusters=False)
        current_ned = ned(word_clusters, print_inpure=False)

        i += 1
        print(f"Iteration {i}: NED: {current_ned:.6f}")

        is_ned_improvement = current_ned < best_ned - tolerance
        is_duplicate_improvement = duplicate_counts < best_duplicate_count - 0

        if is_ned_improvement or is_duplicate_improvement:
            best_clusters = word_clusters
            best_ned = current_ned
            best_duplicate_count = duplicate_counts
        else:
            print("Converged early due to no significant improvement in NED or duplicate count.")
            break

    print(f"Best NED: {best_ned:.6f}, Best Duplicates: {best_duplicate_count}")
    return best_ned, best_duplicate_count, best_clusters

def save_cluster_centroids(centroids, dir):
    out_path = Path(dir) / "words.csv"
    centroid_df = pd.DataFrame(columns=["id", "text", "units"])
    for c in range(len(centroids)):
        
        new_row = pd.DataFrame(
            [[c, centroids[c].true_word, centroids[c].clean_encoding]],
            columns=centroid_df.columns,
        )
        centroid_df = pd.concat([centroid_df, new_row], ignore_index=True)
    centroid_df.to_csv(out_path, index=False)
    print(f"Wrote centroids to {out_path}")