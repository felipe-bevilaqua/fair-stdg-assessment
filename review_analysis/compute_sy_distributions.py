"""Reviewer-requested analyses (kept outside the tracked codebase).

1. Target (Y), sensitive attribute (S) and joint P(S, Y) distributions for every
   dataset / generator / fold of synthetic data, with the real training
   partition of each fold as reference, plus mean/std across folds.
2. Approximate training + tuning times per dataset / generator, aggregated from
   the Optuna trial logs in ``results/from_s3`` (column ``running_time`` =
   fit + sampling wall-clock seconds of one trial).

Outputs (written to ``review_analysis/outputs/``):
- sy_distributions_per_fold.csv
- sy_distributions_summary.csv   (mean/std across folds)
- training_times_per_fold.csv
- training_times_summary.csv
- fig_joint_sy_<dataset>.png     (real vs. synthetic joint P(S,Y) per generator)

Conventions follow ``src/utils.run_clf`` / ``configs/datasets_config.json``:
Y = 1 iff target_col == target_value; S = 1 marks the *protected* group,
i.e. col == sensitive_value when sensitive_value_type == 'Protected' and
col != sensitive_value when it is 'Privileged' (COMPAS).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Self-contained mode: use the data/results/configs copies bundled in this
# folder when present (the zipped bundle); otherwise fall back to the repo root.
ROOT = HERE if (HERE / "data" / "synthetic_data").exists() else HERE.parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATASETS = ["adult", "bank_marketing", "compas", "german"]
GENERATORS = ["arf", "ctabgan", "ctgan", "ddpm", "realtabformer", "tvae"]
FOLDS = range(5)

with open(ROOT / "configs" / "datasets_config.json") as f:
    DATASET_CONFIG = json.load(f)


def binarize(series: pd.Series, value) -> pd.Series:
    """Binary indicator series == value, robust to str/numeric dtype drift."""
    if pd.api.types.is_numeric_dtype(series) and isinstance(value, (int, float)):
        return (series == float(value)).astype(int)
    return (series.astype(str).str.strip() == str(value).strip()).astype(int)


def sy_row(df: pd.DataFrame, cfg: dict) -> dict:
    y = binarize(df[cfg["target_col"]], cfg["target_value"])
    s_eq = binarize(df[cfg["sensitive_col"]], cfg["sensitive_value"])
    # S = 1 -> protected group (same convention as run_clf in src/utils.py)
    s = s_eq if cfg["sensitive_value_type"] == "Protected" else 1 - s_eq
    n = len(df)
    row = {
        "n": n,
        "p_y1": y.mean(),
        "p_y0": 1 - y.mean(),
        "p_s_protected": s.mean(),
        "p_s_privileged": 1 - s.mean(),
    }
    for s_val in (0, 1):
        for y_val in (0, 1):
            cnt = int(((s == s_val) & (y == y_val)).sum())
            row[f"count_s{s_val}_y{y_val}"] = cnt
            row[f"p_s{s_val}_y{y_val}"] = cnt / n
    return row


def load_real_train(dataset: str, fold: int) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "original_data" / f"{dataset}.csv")
    with open(ROOT / "data" / "fold_indexes" / f"{dataset}_fold_indexes.json") as f:
        idx = json.load(f)[f"fold_{fold}"]["train"]
    return df.loc[idx]


def compute_distributions() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        cfg = DATASET_CONFIG[dataset]
        for fold in FOLDS:
            rows.append(
                {"dataset": dataset, "generator": "real_train", "fold": fold}
                | sy_row(load_real_train(dataset, fold), cfg)
            )
            for gen in GENERATORS:
                path = (
                    ROOT / "data" / "synthetic_data" / dataset
                    / f"{dataset}_{gen}_fold_{fold}_best.csv"
                )
                if not path.exists():
                    print(f"missing: {path.name}")
                    continue
                # some synthetic CSVs were saved with a pandas index column, some not
                df_syn = pd.read_csv(path)
                df_syn = df_syn.drop(columns=[c for c in df_syn.columns if c.startswith("Unnamed")])
                rows.append(
                    {"dataset": dataset, "generator": gen, "fold": fold}
                    | sy_row(df_syn, cfg)
                )
    return pd.DataFrame(rows)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    prop_cols = [c for c in per_fold.columns if c.startswith("p_")] + ["n"]
    agg = per_fold.groupby(["dataset", "generator"])[prop_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index()


def plot_joint(per_fold: pd.DataFrame) -> None:
    cells = ["p_s0_y0", "p_s0_y1", "p_s1_y0", "p_s1_y1"]
    labels = [
        "P(S=priv, Y=0)", "P(S=priv, Y=1)",
        "P(S=prot, Y=0)", "P(S=prot, Y=1)",
    ]
    order = ["real_train"] + GENERATORS
    for dataset in DATASETS:
        sub = per_fold[per_fold["dataset"] == dataset]
        mean = sub.groupby("generator")[cells].mean().reindex(order)
        std = sub.groupby("generator")[cells].std().reindex(order)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(order))
        width = 0.2
        for i, (cell, label) in enumerate(zip(cells, labels)):
            ax.bar(
                x + (i - 1.5) * width, mean[cell], width,
                yerr=std[cell], capsize=3, label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=20)
        ax.set_ylabel("proportion (mean ± std over 5 folds)")
        ax.set_title(f"Joint P(S, Y): real training partition vs. synthetic — {dataset}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"fig_joint_sy_{dataset}.png", dpi=150)
        plt.close(fig)


def compute_times_per_trial() -> pd.DataFrame:
    """One row per Optuna trial: dataset, generator, fold, trial, running_time."""
    rows = []
    for path in sorted((ROOT / "results" / "from_s3").glob("result_trials_*.csv")):
        if "error" in path.name:
            continue
        stem = path.stem.replace("result_trials_", "")
        dataset = next(d for d in DATASETS if stem.startswith(d))
        rest = stem[len(dataset) + 1:]
        gen, fold = rest.rsplit("_", 1)
        df = pd.read_csv(path)
        for trial, (rt, mcc) in enumerate(zip(df["running_time"], df["MCC"])):
            rows.append({
                "dataset": dataset,
                "generator": gen,
                "fold": int(fold),
                "trial": trial,
                "running_time_s": rt,
                "running_time_min": rt / 60,
                "MCC": mcc,
            })
    return pd.DataFrame(rows).sort_values(["dataset", "generator", "fold", "trial"])


def compute_times() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted((ROOT / "results" / "from_s3").glob("result_trials_*.csv")):
        if "error" in path.name:
            continue
        stem = path.stem.replace("result_trials_", "")
        dataset = next(d for d in DATASETS if stem.startswith(d))
        rest = stem[len(dataset) + 1:]
        gen, fold = rest.rsplit("_", 1)
        df = pd.read_csv(path)
        rt = df["running_time"].dropna()
        rows.append({
            "dataset": dataset,
            "generator": gen,
            "fold": int(fold),
            "n_trials": len(rt),
            "trial_time_mean_s": rt.mean(),
            "trial_time_median_s": rt.median(),
            "trial_time_min_s": rt.min(),
            "trial_time_max_s": rt.max(),
            "total_tuning_time_s": rt.sum(),
            "best_trial_time_s": df.loc[df["MCC"].idxmax(), "running_time"],
        })
    per_fold = pd.DataFrame(rows).sort_values(["dataset", "generator", "fold"])
    summary = (
        per_fold.groupby(["dataset", "generator"])
        .agg(
            folds=("fold", "count"),
            n_trials_total=("n_trials", "sum"),
            trial_time_mean_s=("trial_time_mean_s", "mean"),
            trial_time_median_s=("trial_time_median_s", "mean"),
            trial_time_min_s=("trial_time_min_s", "min"),
            trial_time_max_s=("trial_time_max_s", "max"),
            total_tuning_time_s=("total_tuning_time_s", "sum"),
            best_trial_time_mean_s=("best_trial_time_s", "mean"),
        )
        .reset_index()
    )
    summary["total_tuning_time_h"] = summary["total_tuning_time_s"] / 3600
    return per_fold, summary


if __name__ == "__main__":
    per_fold = compute_distributions()
    per_fold.to_csv(OUT_DIR / "sy_distributions_per_fold.csv", index=False)
    summary = summarize(per_fold)
    summary.to_csv(OUT_DIR / "sy_distributions_summary.csv", index=False)
    plot_joint(per_fold)

    times_per_trial = compute_times_per_trial()
    times_per_trial.to_csv(OUT_DIR / "training_times_per_trial.csv", index=False)
    times_per_fold, times_summary = compute_times()
    times_per_fold.to_csv(OUT_DIR / "training_times_per_fold.csv", index=False)
    times_summary.to_csv(OUT_DIR / "training_times_summary.csv", index=False)

    print(f"wrote outputs to {OUT_DIR}")
    print(per_fold.head())
    print(times_summary)
