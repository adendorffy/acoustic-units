#!/bin/bash

ALIGNMENTS_DIR="librispeech/alignments"
AUDIO_DIR="librispeech/audio"
FEATURES_DIR="features"
OUTPUT_DIR="output"
MODEL="WAVLM_BASE"

LAYERS=(7 8 10 12)  

for LAYER in "${LAYERS[@]}"
do
    echo "---------------------------------------------"
    echo "Processing model: $MODEL at layer $LAYER"
    
    python train_kmeans.py "$MODEL" "$LAYER"

    python encode_my_features.py "$MODEL" "$LAYER"

    python calculate_my_distances.py "$MODEL" "$LAYER" "$FEATURES_DIR" "$OUTPUT_DIR"

    echo "✅ Finished processing model=$MODEL, Layer=$LAYER"
done
