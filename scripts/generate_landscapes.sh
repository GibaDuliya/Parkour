#!/usr/bin/env bash
# Generate N valid landscapes (those where ≥50% of cells can reach the goal).
# Tries seeds 0, 1, 2, ... until N valid landscapes are saved.
# Usage: bash scripts/generate_landscapes.sh [N]
# Example: bash scripts/generate_landscapes.sh 5

N=${1:-5}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Generating $N valid landscapes..."

saved=0
seed=0

while [ "$saved" -lt "$N" ]; do
    output=$(python "$PROJECT_ROOT/run/run_landscape.py" --seed "$seed" 2>&1)
    echo "$output"
    if echo "$output" | grep -q "Saved to:"; then
        saved=$((saved + 1))
        echo "  [$saved/$N] saved (seed=$seed)"
    fi
    seed=$((seed + 1))
done

echo "Done. Generated $N valid landscapes (tried $seed seeds)."
