from pathlib import Path
from utils.features import DataSet
import numpy as np
from utils.features import WordUnit
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import time
import editdistance
from line_profiler import profile


def pair_generator(num_paths):
    for i in range(num_paths):
        for j in range(i + 1, num_paths):
            yield i, j

def get_batch_of_paths(num_paths, chunk_limit=100):
    """Generate sequential batches of (i, j) path pairs."""
    pairs = pair_generator(num_paths) 
    chunk = [] 

    for idx, (i, j) in enumerate(pairs, 1):
        chunk.append((i, j))

        if idx % chunk_limit == 0:
            yield chunk 
            chunk = [] 

    if chunk:  
        yield chunk


@profile
def load_units(path):
    return np.load(path)

@profile
def calculate_distance_per_chunk(chunk_pair):
    
    id_1, id_2 = tuple(chunk_pair.keys())[0]
    feature_1, feature_2 = tuple(chunk_pair.values())[0]

    length = np.max([len(feature_1), len(feature_2)])

    dist = 0
    if length > 0:
        dist =  editdistance.eval(feature_1, feature_2) / length

    return (id_1, id_2, dist)

def info_to_csv(csv_path, file_map):
    rows = [
        (file, file_map[file])
        for file in file_map
    ]
    df = pd.DataFrame(rows, columns=["id", "filename"])
    df.to_csv(csv_path, index=False)

@profile
def main():

    feat_dir = Path("features/dusted_units/0.2/")
    
    file_map = {}
    features = []
    for i, feature in enumerate(feat_dir.rglob("**/*.npy")):
        file_map[i] = feature
        features.append(load_units(feature))

    sample_size = i
    print(f"sample_size: {sample_size}")
    dist_mat = np.zeros((sample_size, sample_size), dtype=np.float32)

    
    csv_path = "output/dusted/info.csv"
    dist_mat_out_path = Path("output/dusted/dist_mat.npz")
    dist_mat_out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_limit = 1000000
    num_pairs = sample_size * (sample_size - 1) // 2
    num_chunks = (num_pairs + chunk_limit - 1) // chunk_limit 

    for chunk in tqdm(get_batch_of_paths(sample_size, chunk_limit=chunk_limit), total=num_chunks, desc="Processing chunks"):
        chunk_units = [{(i,j) : (features[i], features[j])} for i, j in chunk]

        with Pool(7) as pool:
            chunk_results = pool.map(calculate_distance_per_chunk, chunk_units)
        
        for i,j,dist in chunk_results:
            dist_mat[i, j] = dist
        
    
    info_to_csv(file_map, csv_path)
    np.savez_compressed(dist_mat, dist_mat_out_path)

if __name__ == "__main__":
    main()