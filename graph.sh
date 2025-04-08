#!/bin/bash

MODEL="wavlm_base"
LAYER=7
KMEANS_DATA_DIR="librispeech/train-clean-100"
AUDIO_EXT=".flac"
DATA_DIR="librispeech/dev-clean"
ALIGN_DIR="librispeech/alignments/dev-clean"
N_CLUSTERS=(50 100 200 300 400 500)
# N_CLUSTERS=(200)
GAMMA=0.0
THRESHOLD=0.4
RESOLUTION=0.5

for n in "${N_CLUSTERS[@]}"; do


    echo "🔗 Creating graph from distances..."
    python graph.py "$MODEL" "$LAYER" "$GAMMA" "$n" "$ALIGN_DIR" "$THRESHOLD" 

    echo "🧩 Performing clustering..."
    python cluster.py "$MODEL" "$LAYER" "$GAMMA" "$n" "features" "$THRESHOLD" "$RESOLUTION"

    echo "📊 Evaluating clustering results..."
    python evaluate.py "$MODEL" "$LAYER" "$GAMMA" "$n" "graphs" "$THRESHOLD" 


    echo "✅ Completed iteration with N_CLUSTERS = $n!"

    echo "🔄 Committing changes to Git..."
    git add .
    git commit -m "Completed iteration with N_CLUSTERS = $n and GAMMA = $GAMMA"
    git push
done

echo "🎉 All experiments completed successfully!"
