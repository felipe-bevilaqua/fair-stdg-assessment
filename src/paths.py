"""Centralised filesystem paths for the project.

Every path is derived from the repository root. By default the root is the
parent of this file's directory (i.e. the repo root when running from a clone).
Set the ``MSC_ROOT`` environment variable to point the pipeline somewhere else.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("MSC_ROOT", Path(__file__).resolve().parents[1]))

# Data
DATA_DIR = PROJECT_ROOT / "data"
ORIGINAL_DATA_DIR = DATA_DIR / "original_data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic_data"
FOLD_INDEXES_DIR = DATA_DIR / "fold_indexes"

# Configs and saved models
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
FAIR_EXP_DIR = RESULTS_DIR / "fair_exp"

# Vendored CTAB-GAN+ (added to sys.path so `import model...` resolves)
CTABGAN_DIR = PROJECT_ROOT / "CTAB-GAN-Plus"
