from pathlib import Path
from utils.features import DataSet, WordUnit
from encode import sample_files
import numpy as np
import pandas as pd
import editdistance
from tqdm import tqdm
import concurrent.futures
 
def load_units(dataset, sampled_paths, gamma):
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")
    word_id = 0
    
    if gamma == 0.0:
        hubert_words = []
        for path in tqdm(sampled_paths, desc="Loading Units"):
        
            hubert_path = dataset.feat_dir /"hubert_units"
            hubert_paths = list(hubert_path.rglob(f"**/{path.stem}_*.npy"))
            
            for w, h_path in enumerate(hubert_paths):
                
                parts = h_path.stem.split("_")
                h_df = align_df[align_df["filename"]==parts[0]]
                h_df = h_df[h_df["word_id"]==int(parts[1])]

                units = np.load(h_path)

                if not isinstance(h_df['text'].iloc[0], str):
                    true_word = '_'
                else:
                    true_word = h_df['text'].iloc[0]

                word = WordUnit(
                    id=word_id,
                    filename=parts[0], 
                    index=parts[1],
                    true_word=true_word,
                    boundaries=[h_df["word_start"].iloc[0], h_df["word_end"].iloc[0]],
                )

                word.update_encoding(units)
                word_id += 1
                hubert_words.append(word)
        return hubert_words
    else:
        dusted_words = []
        for path in tqdm(sampled_paths, desc="Loading Units"):
        
            dusted_path = dataset.feat_dir /"dusted_units"/str(gamma)
            dusted_paths = list(dusted_path.rglob(f"**/{path.stem}_*.npy"))

            for w, d_path in enumerate(dusted_paths):
                parts = d_path.stem.split("_")
                d_df = align_df[align_df["filename"]==parts[0]]
                d_df = d_df[d_df["word_id"]==int(parts[1])]
                units = np.load(d_path)

                if not isinstance(d_df['text'].iloc[0], str):
                    true_word = '_'
                else:
                    true_word = d_df['text'].iloc[0]

                word = WordUnit(
                    id=w,
                    filename=parts[0], 
                    index=parts[1],
                    true_word=true_word,
                    boundaries=[d_df["word_start"].iloc[0], d_df["word_end"].iloc[0]],
                )

                word.update_encoding(units)
                word_id += 1
                dusted_words.append(word)

        return dusted_words

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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_hubert = executor.submit(load_units, dataset, sampled_paths, 0.0)
        future_dusted = executor.submit(load_units, dataset, sampled_paths, 0.2)

        hubert_words = future_hubert.result()
        dusted_words = future_dusted.result()

    num_words = len(hubert_words)
    true_words = []

    avg_words = []
    for w in tqdm(range(num_words), desc="Calculating Avg words"):
        new_word = hubert_words[w].copy()
        print(new_word.id)
        hubert_units = hubert_words[w].clean_encoding
        dusted_units = dusted_words[w].clean_encoding

        common_units = list(set(hubert_units).intersection(dusted_units))
        new_word.update_encoding(common_units)

        avg_words.append(new_word)
        if not isinstance(new_word.true_word, str):
            true_words.append("_")
        else:
            true_words.append(new_word.true_word)

    print(true_words)


    # dist_mat = np.zeros((num_words, num_words))
    # for i in tqdm(range(num_words), desc="Calculating Distances"):
    #     for j in range(i+1, num_words):
    #         # dist_hubert = editdistance.eval(hubert_words[i].clean_encoding, hubert_words[j].clean_encoding)
    #         # dist_dusted = editdistance.eval(dusted_words[i].clean_encoding, dusted_words[j].clean_encoding)

    #         dist_mat[i, j] = editdistance.eval(avg_words[i].clean_encoding, avg_words[j].clean_encoding)
    #         dist_mat[j, i] = dist_mat[i,j]

    # print(dist_mat[0:5, 0:5])
    # out_dir = Path("output/avg_words")
    # out_dir.mkdir(parents=True, exist_ok=True)
    # out_file = out_dir / "dist_mat.npy"
    # np.save(out_file, dist_mat)
