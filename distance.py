import editdistance
import pandas as pd
from utils.features import load_units_for_chunk
import torch
import time 

def compute_distance(encoding_i, encoding_j):
    shapes = torch.tensor([encoding_i.size()[0], encoding_j.size()[0]]).cuda()
    length = torch.max(shapes)
    
    if length > 0:
        return editdistance.eval(encoding_i, encoding_j) / length
    
    return 0


def calculate_distance_per_chunk(chunk_words):
    """Process sub-chunk and return computed distances with indices"""
    results = []

    for pair in chunk_words:
        encoding_i = torch.from_numpy(pair[0].clean_encoding).cuda()
        encoding_j = torch.from_numpy(pair[1].clean_encoding).cuda()

        distance = compute_distance(encoding_i, encoding_j)
        results.append((pair[0].id, pair[1].id, distance)) 

    return results


def process_chunk(chunk, sampled_paths, dataset, gamma):
    # start_time = time.perf_counter()

    chunk_paths = [{i: sampled_paths[i], j: sampled_paths[j]} for i, j in chunk]
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")
    
    # load_chunk_start = time.perf_counter()
    chunk_words = load_units_for_chunk(dataset, "dusted", chunk_paths, gamma=gamma, align_df=align_df)
    # load_chunk_end = time.perf_counter()
    # print(f"load_chunk execution time: {load_chunk_end - load_chunk_start:.4f} sec")

    chunk_result = calculate_distance_per_chunk(chunk_words)

    # end_time = time.perf_counter()
    # print(f"process_chunk execution time: {end_time - start_time:.4f} sec")
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
