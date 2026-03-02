#!/bin/bash

# pack_for_colab.sh
# Run this from the project root: ./scripts/pack_for_colab.sh

# Get the project root (one level up from where this script lives)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ARCHIVE_NAME="neural_unmixer_colab.zip"

cd "$PROJECT_ROOT"

echo "📦 Packing Neural Un-Mixer + OTT Dataset into root folder..."

# We exclude everything we DON'T want. 
# We specifically list the dataset subfolders to exclude so 'ott' stays.
zip -r "$ARCHIVE_NAME" . \
    -x "*.pt" \
    -x "*.log" \
    -x "*.pyc" \
    -x "__pycache__/*" \
    -x ".git/*" \
    -x "venv/*" \
    -x ".ipynb_checkpoints/*" \
    -x "checkpoints/*" \
    -x "logs/*" \
    -x "results/*" \
    -x ".DS_Store" \
    -x "dataset/reverb/*" \
    -x "dataset/operator/*" \
    -x "dataset/saturator/*" \
    -x "dataset/eq8/*" \
    -x "dataset/ott-deprecated/*"

echo "------------------------------------------"
echo "✅ Created $ARCHIVE_NAME in $(pwd)"
echo "📂 Contents of dataset/ in zip:"
unzip -l "$ARCHIVE_NAME" "dataset/*" | grep "/"