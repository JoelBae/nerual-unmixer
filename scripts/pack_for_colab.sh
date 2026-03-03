#!/bin/bash

# pack_for_colab.sh
# Run from project root: ./scripts/pack_for_colab.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ARCHIVE_NAME="neural_unmixer_colab.zip"

cd "$PROJECT_ROOT"

echo "📦 Packing ALL source code + dataset/ott_retrain/ott/ ..."

# 1. Clean up old zip
rm -f "$ARCHIVE_NAME"

# 2. Step 1: Zip EVERYTHING in the project...
# But EXCLUDE the entire dataset folder and standard junk.
# This ensures models/, src/, utils/, etc. are all caught.
zip -r "$ARCHIVE_NAME" . \
    -x "dataset/*" \
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
    -x "scripts/$ARCHIVE_NAME"

# 3. Step 2: Manually inject ONLY the OTT retrain folder
if [ -d "dataset/ott_retrain/ott" ]; then
    echo "📥 Injecting OTT retrain data..."
    zip -ur "$ARCHIVE_NAME" dataset/ott_retrain/ott/
else
    echo "❌ ERROR: Path 'dataset/ott_retrain/ott' not found!"
    exit 1
fi

# 4. Step 3: Inject the checkpoint so we can resume
if [ -f "checkpoints/ott_proxy.pt" ]; then
    echo "📥 Injecting checkpoints/ott_proxy.pt to allow resuming..."
    zip -ur "$ARCHIVE_NAME" checkpoints/ott_proxy.pt
fi

echo "------------------------------------------"
echo "✅ Created $ARCHIVE_NAME in root."
echo "📊 Final Zip Size: $(du -h "$ARCHIVE_NAME" | cut -f1)"

echo "📂 Verification of Zip Structure:"
unzip -l "$ARCHIVE_NAME" | grep -E "/$" | head -n 15