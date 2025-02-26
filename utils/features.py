from pathlib import Path
import numpy as np 
import pandas as pd
import ace_tools_open as tools
from collections import Counter
from tqdm import tqdm
import torch

class DataSet:
    def __init__(self, name, in_dir, align_dir, feat_dir, audio_ext):
        self.name = name
        self.in_dir = in_dir
        self.align_dir = align_dir
        self.feat_dir = feat_dir

        self.audio_ext = audio_ext
        
        current_dir = Path.cwd()
        self.output_dir = current_dir / "output" 

class WordUnit:
    def __init__(self, id, filename, index, true_word, boundaries, discrete=True):
        self.filename = filename
        self.index = index
        self.discrete = discrete

        self.true_word = true_word
        self.word_boundaries = self.boundary_frames(boundaries)

        self.original_encoding = None
        self.clean_encoding = []
        self.flags = None
        self.id = id
        self.cluster_id = None

    def get_frame_num(self, timestamp, sample_rate, frame_size_ms):
        hop = frame_size_ms/1000 * sample_rate
        hop_size = np.max([hop, 1])
        return int((timestamp * sample_rate) / hop_size)
    
    def boundary_frames(self, boundaries):
        start_frame = self.get_frame_num(boundaries[0], 16000, 20)
        end_frame = self.get_frame_num(boundaries[1], 16000, 20)
        return [start_frame, end_frame]
    
    def add_cluster_id(self, id):
        self.cluster_id = id
    
    def add_encoding_by_flags(self, encoding, flags, discrete):
        if not discrete:
            encoding = encoding.squeeze(0)

        start_frame = self.word_boundaries[0]
        end_frame = self.word_boundaries[1]

        cut_encoding = encoding[start_frame: end_frame]
        cut_flags = flags[start_frame: end_frame]

        self.original_encoding = cut_encoding
        self.flags = cut_flags

        
        for i in range(min(len(self.original_encoding), len(self.flags))):
            if cut_flags[i]:
                if not discrete:
                    self.clean_encoding.append(self.original_encoding[i,:].unsqueeze(0))
                else:
                    self.clean_encoding.append(self.original_encoding[i])

        if not discrete and isinstance(self.clean_encoding, list):
            if self.clean_encoding != []:
                self.clean_encoding = torch.cat(self.clean_encoding, dim=0)
            
    def update_encoding(self, encoding):
        self.clean_encoding = encoding
    
    def copy(self):
        word = WordUnit(
            id=self.id,
            filename=self.filename,
            index=self.index, 
            true_word=self.true_word, 
            boundaries=self.word_boundaries,
            discrete=self.discrete
        )
        word.update_encoding(self.clean_encoding)
        return word
    
def display_words(word_units):

    num_words = len(word_units)
    true_words = []

    for w in range(num_words): 

        if not isinstance(word_units[w].true_word, str):
            true_words.append("_")
        else:
            true_words.append(word_units[w].true_word)
            

    counts = Counter(true_words)

    word_counts_df = pd.DataFrame(counts.items(), columns=["Word", "Count"])
    word_counts_df = word_counts_df.sort_values(by="Count", ascending=False)
    tools.display_dataframe_to_user(name="Sorted Word Counts", dataframe=word_counts_df)

    return true_words

def store_words(clusters, dir):
    out_path = Path(dir) / f"words.csv"
    words_df = pd.DataFrame(columns=['id', 'filename', 'index', 'cluster_id'])
    for c in clusters:
        for word in c:
            new_row = pd.DataFrame([[word.id, word.filename, word.index, word.cluster_id]], columns=words_df.columns)
            words_df = pd.concat([words_df, new_row], ignore_index=True)
    words_df.to_csv(out_path, index=False)
    print(f"Wrote words to {out_path}")

def get_words_and_dist_mat(dataset, model, out_dir, sample_size, gamma=None):

    out_dir = Path(f"output/{model}/{sample_size}")
   
    dist_mat = np.load(out_dir / "dist_mat.npy")
    words_csv = pd.read_csv(out_dir / "words.csv")

    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")

    words = []
    for index, row in tqdm(words_csv.iterrows(), desc=f"Getting {model} words"):
        id = row['id']
        filename = row['filename']
        index = row['index']

        filename_parts = filename.split("-")
        word_path = dataset.feat_dir / f"{model}_units" / filename_parts[0] / filename_parts[1] / f"{filename}_{index}.npy" 
        if gamma:
            word_path = dataset.feat_dir / f"{model}_units" / str(gamma) /filename_parts[0] / filename_parts[1] / f"{filename}_{index}.npy" 

        word = load_word(word_path, id, align_df, Path(dataset.output_dir / model / str(sample_size)))
        words.append(word)
    
    return words, dist_mat

def load_word(word_path, word_id, align_df, from_output=False):
    units = np.load(word_path)
    parts = word_path.stem.split("_")
    word_df = align_df[align_df["filename"]==parts[0]]
    word_df = word_df[word_df["word_id"]==int(parts[1])]

    if not isinstance(word_df['text'].iloc[0], str):
        true_word = "_"
    else:
        true_word = word_df["text"].iloc[0]

    word = WordUnit(
        id=word_id,
        filename=parts[0], 
        index=parts[1],
        true_word=true_word, 
        boundaries=[word_df["word_start"].iloc[0], word_df["word_end"].iloc[0]]
    ) 

    if from_output:
        words_df = pd.read_csv(Path(from_output / "words.csv"))
        word_df = words_df[words_df['filename']==parts[0]]
        word_df = words_df[words_df['id']==int(parts[1])]
        if "cluster_id" in word_df.columns:
            word.add_cluster_id(word_df["cluster_id"].iloc[0])

    word.update_encoding(units)
    return word

def load_units_from_paths(dataset, model, sampled_paths, gamma=None):
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")
    
    words = []
    word_id = 0
    for path in tqdm(sampled_paths, desc="Loading Units"):
        
        model_path  = dataset.feat_dir / f"{model}_units"
        if gamma:
            model_path = dataset.feat_dir / f"{model}_units" / str(gamma)
        
        model_paths = list(model_path.rglob(f"**/{path.stem}_*.npy"))
            
        for m_path in model_paths:
            
            word = load_word(m_path, word_id, align_df)
            words.append(word)
            word_id += 1 

    return words