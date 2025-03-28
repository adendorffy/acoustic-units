GAMMA=${1}
LAYER=7
ALIGNMENTS_DIR="librispeech/alignments"
AUDIO_DIR="librispeech/audio"
FEATURES_DIR="features"
OUTPUT_DIR="output"

echo "Processing Gamma: $GAMMA"

echo "Step 1: Encoding features for Gamma=$GAMMA, Layer=$LAYER"
python encode_features.py "$GAMMA" "$LAYER" "$AUDIO_DIR" "$ALIGNMENTS_DIR" "$FEATURES_DIR"

echo "Step 2: Computing pairwise distances for Gamma=$GAMMA, Layer=$LAYER"
python calculate_distances.py "$GAMMA" "$LAYER" "$FEATURES_DIR" "$OUTPUT_DIR"

echo "Finished processing GAMMA: $GAMMA"