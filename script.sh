#!/bin/bash

MODEL="wavlm_large"
LAYER=11
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS=50
GAMMA=0.5

echo "🔍 Extracting features from training data for k-means clustering..."
python extract_features.py "$KMEANS_DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

echo "📊 Running k-means clustering on extracted features..."
python kmeans.py "$KMEANS_DATA_DIR" "$MODEL" "$LAYER" "$N_CLUSTERS"

echo "🔍 Extracting features from development data for encoding..."
python extract_features.py "$DATA_DIR" "$AUDIO_EXT" "$MODEL" "$LAYER"

echo "🎯 Encoding development data using learned clusters..."
python encode_features.py "$DATA_DIR" "$ALIGN_DIR" "$MODEL" "$LAYER" "$GAMMA" "$N_CLUSTERS"

echo "📏 Computing distances between encoded segments with gamma=$GAMMA..."
python distance.py "$MODEL" "$LAYER" "$GAMMA" "$N_CLUSTERS" 

echo "✅ All steps completed successfully!"
