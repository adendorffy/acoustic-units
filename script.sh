#!/bin/bash

MODEL="wavlm_base"
LAYER=7
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS=(200)
THRESHOLDS=(0.2 0.4 0.5 0.7)
GAMMA=0.2

for n in "${N_CLUSTERS[@]}"; do
    echo "🔍 Extracting features from development data for encoding..."
    python extract_features.py "$DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

    echo "🎯 Encoding development data using learned clusters..."
    python encode_features.py "$DATA_DIR" "$ALIGN_DIR" "$MODEL" "$LAYER" "$GAMMA" "$n"
    
    echo "📏 Calculating distances between segments..."
    python distance.py "$MODEL" "$LAYER" "$GAMMA" "$n"

    for t in "${THRESHOLDS[@]}"; do
        echo "🔗 Creating graph from distances with threshold $t..."
        python graph.py "$MODEL" "$LAYER" "$GAMMA" "$n" "$ALIGN_DIR" "$t" 

        echo "🔄 Committing changes to Git..."

        git add .
        git commit -m "Completed iteration with N_CLUSTERS = $n and GAMMA = $GAMMA and THRESHOLD = $t"
        git push
    done

done

echo "🎉 All steps completed successfully!"