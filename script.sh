
MODEL="wavlm_large"
LAYER=11
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS=500
GAMMA=0.5

python extract_features.py "$KMEANS_DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER" 

python kmeans.py "$KMEANS_DATA_DIR" "$MODEL" "$LAYER"  "$N_CLUSTERS"

python extract_features.py "$DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

python encode_features.py "$DATA_DIR" "$ALIGN_DIR" "$MODEL" "$LAYER" "$N_CLUSTERS"

python distance.py  "$MODEL" "$LAYER" "$GAMMA" 