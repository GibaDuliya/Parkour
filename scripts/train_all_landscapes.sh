#!/usr/bin/env bash
# Train an algorithm on all landscapes found in the landscape/ folder.
# Usage: bash scripts/train_all_landscapes.sh <algorithm>
# Example: bash scripts/train_all_landscapes.sh value_iteration

ALGORITHM=${1:?"Usage: $0 <algorithm>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LANDSCAPE_DIR="$PROJECT_ROOT/landscape"

if [ ! -d "$LANDSCAPE_DIR" ]; then
    echo "Error: landscape directory not found at $LANDSCAPE_DIR"
    exit 1
fi

LANDSCAPES=( $(ls -d "$LANDSCAPE_DIR"/landscape_* 2>/dev/null | sort -V) )

if [ ${#LANDSCAPES[@]} -eq 0 ]; then
    echo "Error: no landscape folders found in $LANDSCAPE_DIR"
    exit 1
fi

echo "Found ${#LANDSCAPES[@]} landscapes. Training '$ALGORITHM'..."

for LPATH in "${LANDSCAPES[@]}"; do
    LNAME=$(basename "$LPATH")
    LID="${LNAME#landscape_}"
    echo "  [$LID / $((${#LANDSCAPES[@]} - 1))] Training on $LNAME..."
    python "$PROJECT_ROOT/run/train.py" "$ALGORITHM" --landscape_id "$LID"
done

echo "Done. Results saved to $PROJECT_ROOT/agents/$ALGORITHM/"
