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

# 1. Create the virtual environment (uv downloads Python 3.10 if needed).
uv venv --python 3.10

# 2. Install the pinned dependencies.
uv pip install -r requirements.txt

# 3. Create the directory scaffold the pipeline writes to.
mkdir -p data/original_data data/synthetic_data data/fold_indexes
mkdir -p results/plots
mkdir -p models

echo
echo "Environment ready. Activate it with:  source .venv/bin/activate"
echo
echo "NOTE: the pipeline scripts use absolute '/home/msc/...' paths. Edit those"
echo "paths (or run from /home/msc) before executing the steps in the README."
