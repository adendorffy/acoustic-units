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
def load_word(word_path, word_id, align_df):

    """Loads a word unit with metadata and encoding information."""
    # Load encoding units
    units = np.load(word_path)
    
    # Extract filename and word index
    parts = word_path.stem.split("_")
    filename, index = parts[0], int(parts[1])

    # Filter align_df once using .query()
    word_df = align_df.query("filename == @filename and word_id == @index")
    
    if word_df.empty:
        return None  # Early exit if word not found

    # Extract the actual word text efficiently
    true_word = word_df["text"].iat[0] if isinstance(word_df["text"].iat[0], str) else "_"

    # Create WordUnit object
    word = WordUnit(
        id=word_id,
        filename=filename,
        index=index,
        true_word=true_word,
        boundaries=[word_df["word_start"].iat[0], word_df["word_end"].iat[0]],
    )

    # Update encoding with loaded units
    word.update_encoding(units)
    return word

@profile
def process_key(key, file_map, words_cache, keys, align_df):
    """Helper function to process a single key."""
    if key in words_cache:
        return words_cache[key]  # Retrieve from cache
    
    path = file_map.get(key)
    if path is None:
        print(f"Warning: No file found for key '{key}' in file_map")
        return None  # Skip processing for missing files

    word = load_word(path, key, align_df)  # Load word
    words_cache[key] = word  # Cache it
    keys.add(key)
    return word

@profile
def load_units_for_chunk(chunk, file_map, align_df):
    """Optimized function for loading units for a chunk with parallel loading using joblib."""
    
    words_cache = {}  # Cache for fast word retrieval
    keys = set()
    chunk_words = []

    for chunk_pair in chunk:
        pair_keys = tuple(chunk_pair.keys())

        words= []
        for key in pair_keys:
            words.append(process_key(key, file_map, words_cache, keys, align_df))
            
        chunk_words.append(tuple(words))

    return chunk_words

def store_words_for_chunk(chunk_words, words_df, path):

    new_rows = [
        (word.id, word.filename, word.index, word.cluster_id)
        for pair in chunk_words for word in pair
    ]
    new_df = pd.DataFrame(new_rows, columns=words_df.columns)
    words_df = pd.concat([words_df, new_df], ignore_index=True)

    words_df.to_csv(path, index=False)

@profile
def calculate_distance_per_chunk(chunk_pair):
    """Process chunk-pair and return computed distances with indices"""
    
    encoding_i = chunk_pair[0].clean_encoding
    encoding_j = chunk_pair[1].clean_encoding

    length = np.max([len(encoding_i), len(encoding_j)])

    dist = 0
    if length > 0:
        dist =  editdistance.eval(encoding_i, encoding_j) / length

    return (chunk_pair[0].id, chunk_pair[1].id, dist)

@profile
def main():
    name = "librispeech-dev-clean"
    in_dir = Path("data/dev-clean")
    align_dir = Path("data/alignments/dev-clean")
    feat_dir = Path("features")
    audio_ext = ".flac" 

    dataset = DataSet(
        name, in_dir, align_dir, feat_dir, audio_ext 
    )

    file_map = {}
    for i, feature in enumerate(Path(dataset.feat_dir / "dusted_units/0.2/").rglob("**/*.npy")):
        file_map[i] = feature
        
    sample_size = i
    print(f"Sample size: {i}")

    dist_mat = np.zeros((sample_size, sample_size), dtype=np.float32)
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")

    word_csv_path = "output/dusted/words.csv"
    out_path = Path("output/dusted/dist_mat.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_limit = 1000000
    num_pairs = sample_size * (sample_size - 1) // 2
    num_chunks = (num_pairs + chunk_limit - 1) // chunk_limit 

    words = []
    start_time = time.perf_counter()
    for chunk in tqdm(get_batch_of_paths(sample_size, chunk_limit=chunk_limit), total=num_chunks, desc="Processing chunks"):
        chunk_paths = [{i: file_map[i], j: file_map[j]} for i, j in chunk]
        chunk_words = load_units_for_chunk(chunk_paths, file_map=file_map, align_df=align_df)
        with Pool(7) as pool:
            chunk_results = pool.map(calculate_distance_per_chunk, chunk_words)
        
        words.extend(chunk_words)

        for i, j, dist in chunk_results:
            dist_mat[i, j] = dist
        break
    
   
    # print(f"saving words to {word_csv_path} and dist_mat to {out_path}")
    # words_df = pd.read_csv(word_csv_path)
    # store_words_for_chunk(words, words_df, word_csv_path)
    # np.savez_compressed(out_path, dist_mat.numpy())

    end_time = time.perf_counter()

    print(f"Total time: {end_time - start_time}s")

if __name__ == "__main__":

    main()