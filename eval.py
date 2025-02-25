import itertools
import editdistance
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def ned(word_clusters, print_pure=False):

    distances = []
    for i, clust in enumerate(word_clusters):
        if len(clust)> 1:
            clust_dist = []
           
            for p, q in itertools.combinations(clust, 2):
                dist = editdistance.eval(p.true_word, q.true_word)
                clust_dist.append(dist)
                distances.append(dist)

            if any(dist > 0 for dist in clust_dist) or print_pure:
                print(f"Cluster {i}: {statistics.mean(clust_dist)}")
                for p, q in itertools.combinations(clust, 2):   
                    if editdistance.eval(p.true_word, q.true_word) > 0 or print_pure:
                        print(p.id, q.id, (p.true_word, q.true_word))
                        print(len(clust))

                print()

    return statistics.mean(distances) if distances else 0


def pairwise_edit_dist_mat(dist_mat, title, true_words):

    condensed_dist_mat = squareform(dist_mat)
    dist_df_hub = pd.DataFrame(dist_mat, index=true_words, columns=true_words)
    linked = linkage(condensed_dist_mat, method='average')
    order = leaves_list(linked)
    reordered_dist_df = dist_df_hub.iloc[order, order]

    plt.Figure(figsize=(8,6))
    sns.heatmap(reordered_dist_df, cmap='viridis')
    plt.title(title)
    plt.show()

    
def get_word_clusters(int_clusters, words):

    word_clusters = []
    for clust in int_clusters:
        words_ = []
        for k in range(len(clust)):
            word_k = [w for w in words if w.id == clust[k]]
            words_.append(word_k[0])
        word_clusters.append(words_)
    
    return word_clusters


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
       
    return count/total, total