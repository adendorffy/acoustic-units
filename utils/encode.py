from pathlib import Path
from features import DataSet, WordUnit
import random
from torchaudio.functional import resample
import torch
import torchaudio
from tqdm import tqdm
from webrtcvad import Vad
import struct
import numpy as np
import torch.nn.functional as F
import pandas as pd

INT16_MAX = (2**15) - 1
hop_length = 320
sample_rate = 16000

def sample_files(dataset, sample_size = 100):
    in_paths = list(dataset.in_dir.rglob(f"**/*{dataset.audio_ext}"))

    if sample_size == -1:
        return in_paths

    if sample_size > len(in_paths):
        sample_files = in_paths
    else:
        sample_files = random.sample(in_paths, sample_size)
    return sample_files

def mark_sil(vad, wav):
    wav = F.pad(wav, (40, 40))  
    wav = wav[:, : wav.size(-1) - (wav.size(-1) % hop_length)]

    pcm = struct.pack(
        "%dh" % wav.size(-1),
        *(np.round(wav.squeeze().numpy() * INT16_MAX)).astype(np.int16),
    )

    flags = []
    for window_start in range(0, wav.size(-1), hop_length):
        window_end = window_start + hop_length
        flag = vad.is_speech(pcm[window_start * 2 : window_end * 2], sample_rate)
        flags.append(flag)
    return flags



def get_hubert_units(dataset, sampled_paths):
    words = []

    hubert = torch.hub.load(
        "bshall/hubert:main",
        "hubert_discrete",
        trust_repo=True,
    )
    vad = Vad()

    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")

    for wav_path in tqdm(sampled_paths, desc="Getting HuBERT units"):
        
        wav_df = align_df[align_df['filename'] == wav_path.stem]

        wav, sr = torchaudio.load(wav_path)
        wav = resample(wav, sr, 16000)
        
        flags = mark_sil(vad, wav)
        wav = wav.unsqueeze(0)

        with torch.inference_mode():
            units = hubert.units(wav)
        
        units = units.numpy()

        for w in range(1, max(wav_df['word_id'])):
            word_df = wav_df[wav_df['word_id'] == w]

            new_word = WordUnit(
                filename=wav_path.stem, 
                index=w, 
                true_word=word_df['text'].iloc[0],
                boundaries= [word_df['word_start'].iloc[0], word_df['word_end'].iloc[0]], 
                discrete=True
            )
            
            new_word.add_encoding_by_flags(
                units, flags, discrete=True
            )
            
            words.append(new_word)

            out_path = dataset.feat_dir / "hubert_units" / wav_path.relative_to(dataset.in_dir).with_suffix("")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path = str(out_path) + f"_{w}.npy"
            np.save(out_path, new_word.clean_encoding)

    return words

def get_dusted_units(dataset, sampled_paths, layer=7, gamma=0.2):
    words = []

    kmeans, segment = torch.hub.load(
        "bshall/dusted:main", "kmeans", language="english", trust_repo=True
    )

    hubert, encode = torch.hub.load(
        "bshall/dusted:main", "hubert", language="english", trust_repo=True
    )

    vad = Vad()
    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")

    for wav_path in tqdm(sampled_paths, desc="Getting DUSTED units"):
        
        wav_df = align_df[align_df['filename'] == wav_path.stem]
        wav, sr = torchaudio.load(wav_path)
        wav = resample(wav, sr, 16000)

        flags = mark_sil(vad, wav)
        wav = wav.unsqueeze(0)

        encoding = encode(hubert, wav, layer)
        
        for w in range(max(wav_df['word_id'])):
            word_df = wav_df[wav_df['word_id'] == w]

            new_word = WordUnit(
                filename=wav_path.stem, 
                index=w, 
                true_word=word_df['text'].iloc[0],
                boundaries= [word_df['word_start'].iloc[0], word_df['word_end'].iloc[0]], 
                discrete=True
            )
            
            new_word.add_encoding_by_flags(
                encoding, flags, discrete=False
            )

            if new_word.clean_encoding != []: 
                codes, _ = segment(new_word.clean_encoding, kmeans.cluster_centers_, gamma)

                new_word.update_encoding(codes)
                words.append(new_word)

                out_path = dataset.feat_dir / "dusted_units" / str(gamma) / wav_path.relative_to(dataset.in_dir).with_suffix("")

                out_path.parent.mkdir(parents=True, exist_ok=True)

                out_path = str(out_path) + f"_{w}.npy"
                np.save(out_path, new_word.clean_encoding)

    return words


current_dir = Path.cwd()

dataset = DataSet(
    name="librispeech-dev-clean",
    in_dir=Path("data/dev-clean"),
    align_dir=Path("data/alignments/dev-clean"),
    feat_dir=Path("features"), 
    audio_ext=".flac" 
)

sampled_paths = sample_files(dataset,-1)

hubert_words = get_hubert_units(dataset, sampled_paths)
dusted_words = get_dusted_units(dataset, sampled_paths)
