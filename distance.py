from pathlib import Path
from utils.features import DataSet
from encode import sample_files
import numpy as np
import editdistance
from tqdm import tqdm
from joblib import Parallel, delayed

def compute_distance(args):

    i, j, words = args
    encoding_i = words[i].clean_encoding
    encoding_j = words[j].clean_encoding
    length = max(len(encoding_i), len(encoding_j))
    
    if length > 0:
        return (i, j, editdistance.eval(encoding_i, encoding_j) / length)
    else:
        return (i, j, 0)

def calculate_distance(words, save=None, n_jobs=-1):
    num_words = len(words)
    dist_mat = np.zeros((num_words, num_words))
    
    index_pairs = [(i, j, words) for i in range(num_words) for j in range(i+1, num_words)]

    with tqdm(total=len(index_pairs), desc="Calculating Distances") as pbar:
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(lambda args: (pbar.update(1), compute_distance(args))[1])(args) for args in index_pairs
        )
    for i, j, dist in results:
        dist_mat[i, j] = dist
        dist_mat[j, i] = dist

    print(dist_mat[0:5, 0:5])

    if save:
        Path(save).mkdir(parents=True, exist_ok=True)
        out_file = Path(save) / "dist_mat.npy"
        np.save(out_file, dist_mat)

    return dist_mat

if __name__ == "__main__":

    current_dir = Path.cwd()

    dataset = DataSet(
        name="librispeech-dev-clean",
        in_dir=Path("data/dev-clean"),
        align_dir=Path("data/alignments/dev-clean"),
        feat_dir=Path("features"), 
        audio_ext=".flac" 
    )

    sampled_paths = sample_files(dataset, 4)

    hubert_words, dusted_words = load_units(dataset, sampled_paths, 0.2)

    num_words = len(hubert_words)
    true_words = []

    avg_words = []
    for w in tqdm(range(num_words), desc="Calculating Avg words"):
        new_word = hubert_words[w].copy()
        hubert_units = hubert_words[w].clean_encoding
        dusted_units = dusted_words[w].clean_encoding

        common_units = list(set(hubert_units).intersection(dusted_units))
        new_word.update_encoding(common_units)

        avg_words.append(new_word)
        if not isinstance(new_word.true_word, str):
            true_words.append("_")
        else:
            true_words.append(new_word.true_word)

    out_dir = Path("output/avg_words/")
    dist_mat_dusted = calculate_distance(dusted_words, out_dir)

   
