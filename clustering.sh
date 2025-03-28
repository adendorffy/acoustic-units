GAMMA=${1}
LAYER=7
ALIGNMENTS_DIR="librispeech/alignments"
AUDIO_DIR="librispeech/audio"
FEATURES_DIR="features"
OUTPUT_DIR="output"
NUM_CLUSTERS=13967
THRESHOLD=${2}
RESOLUTION=${3}


echo "Processing Gamma: $GAMMA"

echo "Step 3: Clustering for Gamma=$GAMMA, Layer=$LAYER"

echo "a) Build Graph"
python build_graph.py "$GAMMA" "$LAYER" "$OUTPUT_DIR" --threshold "$THRESHOLD"

echo "b) Calculate Partition"
python cluster.py "$GAMMA" "$LAYER" "$OUTPUT_DIR" --num_clusters "$NUM_CLUSTERS" --threshold "$THRESHOLD" --resolution "$RESOLUTION"

echo "Step 4: Evaluating clustering results for Gamma=$GAMMA, Layer=$LAYER"

echo "a) Convert partition to .list files"
python convert_partition.py "$GAMMA" "$LAYER" "$THRESHOLD" "$OUTPUT_DIR" 

echo "b) Calculate NED for .list output files"
python evaluate.py "$GAMMA" "$LAYER" "$THRESHOLD" "$OUTPUT_DIR" "$ALIGNMENTS_DIR" 

