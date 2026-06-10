#!/usr/bin/env bash
#
# Sets up the Python environment for the project with uv.
#
# Requires uv (https://docs.astral.sh/uv/). Install it with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# The study was developed and run on Python 3.10 (torch==1.13.1 does not
# support newer interpreters), so the virtual environment is pinned to 3.10.

set -euo pipefail

# Resolve the project root (the directory that contains this script) and work
# from there, so setup.sh can be invoked from any location.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

# 1. Create the virtual environment (uv downloads Python 3.10 if needed).
uv venv --python 3.10

# 2. Install the pinned dependencies.
uv pip install -r requirements.txt

# 3. Create the directory scaffold the pipeline writes to.
mkdir -p data/original_data data/synthetic_data data/fold_indexes
mkdir -p results/plots
mkdir -p models

echo
echo "Environment ready (project root: $PROJECT_ROOT)."
echo "Activate it with:  source .venv/bin/activate"
echo
echo "Paths resolve to the project root automatically. To use a different data"
echo "location, export PROJECT_ROOT before running the pipeline, e.g.:"
echo "  export PROJECT_ROOT=/mnt/data"
