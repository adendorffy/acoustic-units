#!/bin/bash

MODEL="wavlm_base"
LAYER=7
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS=(50 100 200 300 400 500)
GAMMA=0.1

for n in "${N_CLUSTERS[@]}"; do
    echo "🔍 Extracting features from training data for k-means clustering..."
    python extract_features.py "$KMEANS_DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

    echo "📊 Running k-means clustering on extracted features..."
    python kmeans.py "$KMEANS_DATA_DIR" "$MODEL" "$LAYER" "$n"

    echo "🔍 Extracting features from development data for encoding..."
    python extract_features.py "$DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

    echo "🎯 Encoding development data using learned clusters..."
    python encode_features.py "$DATA_DIR" "$ALIGN_DIR" "$MODEL" "$LAYER" "$GAMMA" "$n"

    echo "Calculating samediff..."
    python samediff.py "features/$MODEL/layer$LAYER/gamma$GAMMA/k$n" "samediff_output/" "$ALIGN_DIR" 

    echo "✅ Completed iteration with N_CLUSTERS = $n!"

    git add .
    git commit -m "Completed iteration with N_CLUSTERS = $n and GAMMA = $GAMMA"
    git push

done

echo "🎉 All steps completed successfully!"