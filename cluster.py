import numpy as np
import json
from pathlib import Path

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

    for node in range(num_nodes):
        if node not in visited:
            new_cluster = bfs(node)
            clusters.append(new_cluster)
    
    return clusters
