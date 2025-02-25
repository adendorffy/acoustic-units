from pathlib import Path
import numpy as np 
import torch

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

    def get_frame_num(self, timestamp, sample_rate, frame_size_ms):
        hop = frame_size_ms/1000 * sample_rate
        hop_size = np.max([hop, 1])
        return int((timestamp * sample_rate) / hop_size)
    
    def boundary_frames(self, boundaries):
        start_frame = self.get_frame_num(boundaries[0], 16000, 20)
        end_frame = self.get_frame_num(boundaries[1], 16000, 20)
        return [start_frame, end_frame]
    
    def change_id(self, id):
        self.id = id
    
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