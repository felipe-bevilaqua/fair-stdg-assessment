# Fairness Assessment and Mitigation of Synthetic Tabular Data Generation

This repository contains the experimental code for a study on the **group
fairness of machine-learning classifiers trained on synthetic tabular data**.

The central question: when real training data is replaced by synthetic data
produced by modern tabular generators, optionally combined with fairness
mitigation algorithms, what happens to downstream classifier *utility* and
*group fairness*? 

The pipeline tunes a range of generative models, produces
synthetic versions of several benchmark datasets, trains classifiers on them,
and evaluates the resulting fairness/utility trade-off against models trained
on the real data.

## Overview

The experiment runs in four stages:

1. **Download & split** (`src/download_data.py`) — downloads the benchmark
   datasets and builds reproducible **5-fold stratified** `train/val/test`
   index sets, saved as JSON.
2. **Generator tuning** (`src/generators_tuning.py`) — for each
   `(dataset, model, fold)`, runs an **Optuna** hyperparameter search. The
   objective is the **MCC of an LGBM classifier trained on synthetic data and
   evaluated on the real validation fold** (train-on-synthetic, test-on-real).
   Fidelity (MMD) and privacy (DCR) are logged alongside.
3. **Synthetic data generation** (`src/generate_data.py`) — reloads the best
   hyperparameters from the tuning trials and regenerates the synthetic
   dataset for each fold.
4. **Fairness experiment** (`src/classifier_fair_exp.py`) — trains classifiers
   on the synthetic data (with and without fairness mitigation), evaluates
   utility and fairness metrics on the **real test fold**, and compares against
   a baseline trained on real data.

### Datasets

| Dataset          | Target            | Sensitive attribute |
| ---------------- | ----------------- | ------------------- |
| Adult            | `income`          | `sex`               |
| German Credit    | `class-label`     | `sex`               |
| COMPAS           | `two_year_recid`  | `race`              |
| Bank Marketing   | `y`               | `age`               |

### Generative models

`tvae`, `ctgan`, `ddpm`, `arf` (via [synthcity](https://github.com/vanderschaarlab/synthcity)),
`realtabformer`, `be_great`, and `ctabgan` (vendored — see below).

### Fairness mitigation methods

- `lgbm` — unmitigated baseline (LightGBM)
- `threshold_opt` — `fairlearn` `ThresholdOptimizer` (post-processing)
- `exp_grad` — `fairlearn` `ExponentiatedGradient` (in-processing)

### Metrics

Utility: Accuracy, F1, MCC, AUC.
Fairness (`fairlearn`): demographic parity difference, equalized odds
difference, equality of opportunity (TPR difference), predictive equality
(FPR difference).
Quality: MMD (fidelity to real data) and DCR (distance to closest record).

## Repository structure

```
src/
  download_data.py        # stage 1: download datasets + build CV folds
  generators_tuning.py    # stage 2: Optuna HPT per (dataset, model, fold)
  generate_data.py        # stage 3: regenerate best synthetic data
  classifier_fair_exp.py  # stage 4: fairness/utility experiment
  data_generator.py       # unified wrapper over all generative models
  utils.py                # preprocessing, classifiers, metrics, plots
  paths.py                # central filesystem paths (override with MSC_ROOT)
configs/                  # dataset + per-dataset HPT search-space configs
CTAB-GAN-Plus/            # vendored, locally-patched CTAB-GAN+ (see below)
requirements.txt          # pinned dependencies (the source of truth)
setup.sh                  # environment setup via uv
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for environment management.
Install uv, then run:

```bash
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a **Python 3.10** virtual environment (the version the study
was run on — `torch==1.13.1` does not support newer interpreters), installs the
pinned dependencies from `requirements.txt`, and creates the directory scaffold.

> **Dependencies:** the authoritative dependency list is `requirements.txt`.
> `pyproject.toml` is kept only for uv project metadata and does not list the
> runtime dependencies.

## Usage

Run the stages in order (examples use the Adult dataset, fold 0):

```bash
# Stage 1 — download datasets and build fold indexes
python src/download_data.py

# Stage 2 — tune a generator (e.g. CTGAN) for a given dataset/fold
python src/generators_tuning.py --dataset_name adult --model_name ctgan --fold 0 --n_trials 20

# Stage 3 — regenerate the best synthetic data
python src/generate_data.py --dataset_name adult --model_name ctgan --fold 0

# Stage 4 — run the fairness experiment
python src/classifier_fair_exp.py --dataset_name adult --fold 0 --constraint equalized_odds
```

All filesystem paths are defined in `src/paths.py` and resolve relative to the
repository root by default, so the stages run from a fresh clone with no path
editing. To point the pipeline at a different location (e.g. a data volume),
set the `MSC_ROOT` environment variable:

```bash
MSC_ROOT=/mnt/data python src/download_data.py
```

The tuning stage also optionally uploads results to an S3 bucket when
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are set; otherwise that step is
skipped and results are written under `results/` locally.

## Vendored dependency: CTAB-GAN+

`CTAB-GAN-Plus/` is a **locally-modified copy** of
[Team-TUD/CTAB-GAN-Plus](https://github.com/Team-TUD/CTAB-GAN-Plus)
(Zhao et al., *CTAB-GAN+: Enhancing Tabular Data Synthesis*,
[arXiv:2204.00401](https://arxiv.org/abs/2204.00401)). It is vendored rather
than installed because the study relies on local patches (exposing `lr` and
`batch_size`, and a `fit(df, ...)` signature) that are not present upstream.
Upstream does not publish an explicit license; attribution and the licensing
caveat are recorded in `CTAB-GAN-Plus/NOTICE`, and this code is not covered by
the repository's top-level MIT license.
