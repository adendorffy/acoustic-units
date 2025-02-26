from pathlib import Path
import numpy as np 
from collections import Counter
import pandas as pd
import ace_tools_open as tools

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

class DataSet:
    def __init__(self, name, in_dir, align_dir, feat_dir, audio_ext):
        self.name = name
        self.in_dir = in_dir
        self.align_dir = align_dir
        self.feat_dir = feat_dir

        self.audio_ext = audio_ext
        
        current_dir = Path.cwd()
        self.output_dir = current_dir.parent  

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