import torch
import torchaudio
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
import joblib


sr = 16000
audio_dir = Path("librispeech/audio")
align_dir = Path("librispeech/alignments")
feat_dir = Path("kmeans-features/")
audio_ext = ".flac"
layer = 8

audio_paths = list(audio_dir.rglob(f"**/*.{audio_ext}"))
align_df = pd.read_csv(align_dir / "alignments.csv")
bundle = torchaudio.pipelines.HUBERT_BASE
model = bundle.get_model()
model.eval()

all_features = []
for path in audio_paths:
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    with torch.inference_mode():
        features, _ = model.extract_features(waveform, layer=layer)
        encoding = features[-1].squeeze(0)

    all_features.append(encoding)

X = torch.cat(all_features, dim=0).numpy()

n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
joblib.dump(kmeans, f"models/kmeans_hubert_layer{layer}_k{n_clusters}.pkl")
