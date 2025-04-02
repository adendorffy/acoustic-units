#!/bin/bash

ALIGNMENTS_DIR="librispeech/alignments"
AUDIO_DIR="librispeech/audio"
FEATURES_DIR="features"
OUTPUT_DIR="output"
MODEL="HUBERT_BASE"
GAMMA=1.0

LAYERS=(7 10 11 12)  

for LAYER in "${LAYERS[@]}"
do
    echo "---------------------------------------------"
    echo "Processing model: $MODEL at layer $LAYER"
    
    # python train_kmeans.py "$MODEL" "$LAYER"

    python encode_my_features.py "$MODEL" "$GAMMA" "$LAYER"

    python calculate_my_distances.py "$MODEL" "$GAMMA" "$LAYER" "$FEATURES_DIR" "$OUTPUT_DIR"

    echo "✅ Finished processing model=$MODEL, Layer=$LAYER"
done
