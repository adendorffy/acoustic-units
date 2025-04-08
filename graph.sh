#!/bin/bash

MODEL="wavlm_base"
LAYER=7
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS={$2}
GAMMA={$1}
THRESHOLDS=(0.4 0.5 0.6 0.7 0.8 0.9)
RESOLUTION=0.5

for t in "${THRESHOLDS[@]}"; do

    echo "🔗 Creating graph from distances..."
    python graph.py "$MODEL" "$LAYER" "$GAMMA" "$N_CLUSTERS" "$ALIGN_DIR" "$t" 

    echo "🔄 Committing changes to Git..."
    git add .
    git commit -m "Completed graph building with N_CLUSTERS = $N_CLUSTERS, GAMMA=$GAMMA and THRESHOLD = $t"
    git push
done

