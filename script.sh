GAMMAS=(0.01 0.1 0.15 0.2)  
LAYER=7
ALIGNMENTS_DIR="librispeech/alignments"
AUDIO_DIR="librispeech/audio"
FEATURES_DIR="features"
OUTPUT_DIR="output"
NUM_CLUSTERS=13967


for GAMMA in "${GAMMAS[@]}"; do
    echo "Processing Gamma: $GAMMA"

    echo "Step 1: Encoding features for Gamma=$GAMMA, Layer=$LAYER"
    python encode_features.py "$GAMMA" "$LAYER" "$AUDIO_DIR" "$ALIGNMENTS_DIR" "$FEATURES_DIR"

    echo "Step 2: Computing pairwise distances for Gamma=$GAMMA, Layer=$LAYER"
    python calculate_distances.py "$GAMMA" "$LAYER" "$FEATURES_DIR" "$OUTPUT_DIR"

    echo "Step 3: Clustering for Gamma=$GAMMA, Layer=$LAYER"
    python cluster.py "$GAMMA" "$LAYER" "$OUTPUT_DIR" --num_clusters "$NUM_CLUSTERS"

    echo "Step 4: Evaluating clustering results for Gamma=$GAMMA, Layer=$LAYER"

    echo "a) Extract alignmentsa and align to feature paths"
    python extract_alignment.py "$GAMMA" "$LAYER" "$ALIGNMENTS_DIR" "$FEATURES_DIR" "$OUTPUT_DIR"

    echo "b) Convert partition to .list files"
    python convert_partition.py "$GAMMA" "$LAYER" "$ALIGNMENTS_DIR" 

    

done

echo "🎉 All gamma values processed!"