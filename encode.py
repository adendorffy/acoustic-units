from pathlib import Path
from utils.features import DataSet, WordUnit
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
from joblib import Parallel, delayed


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

def get_units(dataset, sampled_paths, gamma=0.2, layer=7,  save=False):

    hubert_words = []
    dusted_words = []

    hubert = torch.hub.load(
        "bshall/hubert:main", "hubert_discrete", trust_repo=True,
    )

    kmeans, segment = torch.hub.load(
        "bshall/dusted:main", "kmeans", language="english", trust_repo=True
    )

    dusted_hubert, encode = torch.hub.load(
        "bshall/dusted:main", "hubert", language="english", trust_repo=True
    )

    vad = Vad()

    align_df = pd.read_csv(dataset.align_dir / "alignments.csv")
    
    word_id_h = 0
    word_id_d = 0
    for wav_path in tqdm(sampled_paths, desc="Getting units"):
        
        wav_df = align_df[align_df['filename'] == wav_path.stem]

        wav, sr = torchaudio.load(wav_path)
        wav = resample(wav, sr, 16000)
        
        flags = mark_sil(vad, wav)
        wav = wav.unsqueeze(0)

        hub_words, word_id_h = get_hubert_units(wav, hubert, wav_path, flags, wav_df, word_id_h, save)
        dust_words, word_id_d = get_dusted_units(wav, dusted_hubert, encode, segment, kmeans, wav_path, flags, wav_df, word_id_d, gamma, layer, save)

        hubert_words.extend(hub_words)
        dusted_words.extend(dust_words)

    return hubert_words, dusted_words

def get_hubert_units(wav, hubert, wav_path, flags, wav_df, word_id, save=False):
    words = []

    with torch.inference_mode():
        units = hubert.units(wav)
    
    units = units.numpy()

    for w in range(max(wav_df['word_id'])):
        
        word_df = wav_df[wav_df['word_id'] == w]
        
        if not isinstance(word_df['text'].iloc[0], str):
            true_word = '_'
        else:
            true_word = word_df['text'].iloc[0]

        new_word = WordUnit(
            id=word_id,
            filename=wav_path.stem, 
            index=w, 
            true_word=true_word,
            boundaries= [word_df['word_start'].iloc[0], word_df['word_end'].iloc[0]], 
            discrete=True
        )
        
        new_word.add_encoding_by_flags(
            units, flags, discrete=True
        )   
        
        word_id += 1
        words.append(new_word)

        if save:
            out_path = dataset.feat_dir / "hubert_units" / wav_path.relative_to(dataset.in_dir).with_suffix("")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path = str(out_path) + f"_{w}.npy"
            np.save(out_path, new_word.clean_encoding)

    return words, word_id

def get_dusted_units(wav, hubert, encode, segment, kmeans, wav_path, flags, wav_df, word_id, gamma=0.2, layer=7, save=False):
    words = []

    encoding = encode(hubert, wav, layer)
    
    for w in range(max(wav_df['word_id'])):
        word_df = wav_df[wav_df['word_id'] == w]

        if not isinstance(word_df['text'].iloc[0], str):
            true_word = '_'
        else:
            true_word = word_df['text'].iloc[0]

        new_word = WordUnit(
            id=word_id,
            filename=wav_path.stem, 
            index=w, 
            true_word=true_word,
            boundaries= [word_df['word_start'].iloc[0], word_df['word_end'].iloc[0]], 
            discrete=True
        )
        word_id += 1
        
        new_word.add_encoding_by_flags(encoding, flags, False)

        if new_word.clean_encoding == []:
            codes = []
        else:
            codes, _ = segment(new_word.clean_encoding, kmeans.cluster_centers_, gamma)

        new_word.update_encoding(codes)
        words.append(new_word)

        if save: 
            out_path = dataset.feat_dir / "dusted_units" / str(gamma) / wav_path.relative_to(dataset.in_dir).with_suffix("")

            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_path = str(out_path) + f"_{w}.npy"
            np.save(out_path, new_word.clean_encoding)


    return words, word_id

if __name__ == "__main__":

    current_dir = Path.cwd()

    dataset = DataSet(
        name="librispeech-dev-clean",
        in_dir=Path("data/dev-clean"),
        align_dir=Path("data/alignments/dev-clean"),
        feat_dir=Path("features"), 
        audio_ext=".flac" 
    )

    sampled_paths = sample_files(dataset,-1)
    
    hubert_words, dusted_words = Parallel(n_jobs=8)(
    [
        delayed(get_units)(dataset, sampled_paths, 0.2, 7, True),  
        delayed(get_units)(dataset, sampled_paths, 0.2, 7, True)   
    ]
)