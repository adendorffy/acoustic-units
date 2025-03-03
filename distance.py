from pathlib import Path
import editdistance
import pandas as pd
from utils.features import load_units_for_chunk

def compute_distance(encoding_i, encoding_j):

    length = max(len(encoding_i), len(encoding_j))
    
    if length > 0:
        return editdistance.eval(encoding_i, encoding_j) / length
    
    return 0


def calculate_distance_per_chunk(chunk_words, dist_mat):
    
    for pair in chunk_words:
        
        encoding_i = pair[0].clean_encoding
        encoding_j = pair[1].clean_encoding
        

        dist_mat[pair[0].id, pair[1].id] = compute_distance(encoding_i, encoding_j)

    return dist_mat 

def process_chunk(chunk, sampled_paths, dataset, gamma, dist_mat):
    chunk_paths = [{i: sampled_paths[i], j: sampled_paths[j]} for i, j in chunk]
    chunk_words = load_units_for_chunk(dataset, "dusted", chunk_paths, gamma)
    chunk_result = calculate_distance_per_chunk(chunk_words, dist_mat)
    return chunk_result, chunk_words

def store_words_for_chunk(chunk_words, path):
    words_df = pd.read_csv(path)
    
    for pair in chunk_words:
        for word in pair:
            new_row = pd.DataFrame(
                [[word.id, word.filename, word.index, word.cluster_id]],
                columns=words_df.columns,
            )
            words_df = pd.concat([words_df, new_row], ignore_index=True)
    words_df.to_csv(path, index=False)
