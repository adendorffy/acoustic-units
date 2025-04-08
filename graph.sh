#!/bin/bash

MODEL="wavlm_base"
LAYER=7
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
# N_CLUSTERS=(50 100 200 300 400 500)
N_CLUSTERS=(200)
GAMMA=0.0
THRESHOLD=0.4
RESOLUTION=1.0

for n in "${N_CLUSTERS[@]}"; do
    # echo "🔍 Extracting features from training data for k-means clustering..."
    # python extract_features.py "$KMEANS_DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

    # echo "📊 Running k-means clustering on extracted features..."
    # python kmeans.py "$KMEANS_DATA_DIR" "$MODEL" "$LAYER" "$n"

    # echo "🔍 Extracting features from development data for encoding..."
    # python extract_features.py "$DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

    # echo "🎯 Encoding development data using learned clusters..."
    # python encode_features.py "$DATA_DIR" "$ALIGN_DIR" "$MODEL" "$LAYER" "$GAMMA" "$n"

    # echo "📏 Calculating distances between segments..."
    # python distance.py "$MODEL" "$LAYER" "$GAMMA" "$n"

    echo "🔗 Creating graph from distances..."
    python graph.py "$MODEL" "$LAYER" "$GAMMA" "$n"

    echo "🧩 Performing clustering..."
    python cluster.py "$MODEL" "$LAYER" "$GAMMA" "$n" "features" "$THRESHOLD" "$RESOLUTION"

    echo "📊 Evaluating clustering results..."
    python evaluate.py "$MODEL" "$LAYER" "$GAMMA" "$n" "features" "$THRESHOLD" "$RESOLUTION"

    echo "🔁 Calculating SameDiff score..."
    python samediff.py "features/$MODEL/layer$LAYER/gamma$GAMMA/k$n" "$ALIGN_DIR"

    echo "✅ Completed iteration with N_CLUSTERS = $n!"

    echo "🔄 Committing changes to Git..."
    git add .
    git commit -m "Completed iteration with N_CLUSTERS = $n and GAMMA = $GAMMA"
    git push
done

echo "🎉 All experiments completed successfully!"
