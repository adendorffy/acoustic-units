from pathlib import Path
from utils.features import DataSet, WordUnit
from encode import sample_files
import numpy as np
import pandas as pd
import editdistance
from tqdm import tqdm
 
def load_units(dataset, sampled_paths, gamma):
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")
    
    hubert_words = []
    dusted_words = []
    word_id_h = 0
    word_id_d = 0

    for path in tqdm(sampled_paths, desc="Loading Units"):
    
        hubert_path = dataset.feat_dir /"hubert_units"
        hubert_paths = list(hubert_path.rglob(f"**/{path.stem}_*.npy"))
        

        dusted_path = dataset.feat_dir /"dusted_units"/str(gamma)
        dusted_paths = list(dusted_path.rglob(f"**/{path.stem}_*.npy"))
            
        for h_path, d_path in zip(hubert_paths, dusted_paths):
            
            parts = h_path.stem.split("_")

            h_df = align_df[align_df["filename"]==parts[0]]
            h_df = h_df[h_df["word_id"]==int(parts[1])]

            units = np.load(h_path)

            if not isinstance(h_df['text'].iloc[0], str):
                true_word = '_'
            else:
                true_word = h_df['text'].iloc[0]

            hub_word = WordUnit(
                id=word_id_h,
                filename=parts[0], 
                index=parts[1],
                true_word=true_word,
                boundaries=[h_df["word_start"].iloc[0], h_df["word_end"].iloc[0]],
            )

            hub_word.update_encoding(units)
            hubert_words.append(hub_word)

            word_id_h +=1 

            d_df = align_df[align_df["filename"]==parts[0]]
            d_df = d_df[d_df["word_id"]==int(parts[1])]
            units = np.load(d_path)

            if not isinstance(d_df['text'].iloc[0], str):
                true_word = '_'
            else:
                true_word = d_df['text'].iloc[0]

            dust_word = WordUnit(
                id=word_id_d,
                filename=parts[0], 
                index=parts[1],
                true_word=true_word,
                boundaries=[d_df["word_start"].iloc[0], d_df["word_end"].iloc[0]],
            )

            dust_word.update_encoding(units)
            dusted_words.append(dust_word)
            word_id_d += 1

    return hubert_words, dusted_words

def calculate_distance(words, save=None):
    num_words = len(words)
    dist_mat = np.zeros((num_words, num_words))
    for i in tqdm(range(num_words), desc="Calculating Distances"):
        encoding_i = words[i].clean_encoding
        for j in range(i+1, num_words):
            encoding_j = words[j].clean_encoding
            length = max(len(encoding_i), len(encoding_j))
            
            if length > 0:
                dist = editdistance.eval(encoding_i, encoding_j) / length
            else:
                dist = 0 
            
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist

    print(dist_mat[0:5, 0:5])
    if save: 
        save.mkdir(parents=True, exist_ok=True)
        out_file = save / "dist_mat.npy"
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

   
