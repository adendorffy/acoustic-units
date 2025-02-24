from pathlib import Path

class DataSet:
    def __init__(self, name, in_dir, align_dir, feat_dir, model, layer, audio_ext):
        self.name = name
        self.in_dir = in_dir
        self.align_dir = align_dir
        self.feat_dir = feat_dir

        self.model = model
        self.layer = layer
        self.audio_ext = audio_ext
        
        current_dir = Path.cwd()
        self.output_dir = current_dir.parent / model / str(layer)

class Cluster:
    def __init__(self, words, id):
        self.words = words if words else []
        self.id = id

class WordUnit:
    def __init__(self, filename, index, encoding, discrete=True):
        self.filename = filename
        self.index = index
        self.encoding = encoding
        self.discrete = discrete
        self.true_word = None
        self.word_boundaries = []
    
    def add_true_word(self, true_word):
        self.true_word = true_word
    
    def add_word_boundaries(self, boundaries):
        self.word_boundaries = boundaries