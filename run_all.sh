#!/bin/sh
# Runs every script and refreshes figures/ and outputs/
cd "$(dirname "$0")/code" || exit 1
for f in Task_*.py; do
    echo "Running $f"
    python3 "$f" > "../outputs/${f%.py}.txt" 2>&1 || echo "  -> $f failed, see outputs/"
done
