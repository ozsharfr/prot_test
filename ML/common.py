"""
common.py — Shared setup, data loading, feature preparation, and plotting
           used by both regressor.py and classifier.py.
"""

import sys
import os
import re
import logging
import warnings
from pathlib import Path

# --- Path setup: must come before any local imports ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

from features import build_features
from config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TARGET = "prediction_error"

ALWAYS_DROP = {
    "#Pdb", "Mutation(s)_PDB", "Mutation(s)_cleaned", "iMutation_Location(s)",
    "Hold_out_type", "Hold_out_proteins", "Reference", "Protein 1", "Protein 2",
    "Notes", "Method", "SKEMPI version", "pdb_id", "chain", "wt_aa", "mut_aa",
    "resnum_str", "has_insertion_code",
    "Affinity_mut (M)", "Affinity_mut_parsed", "Affinity_wt (M)", "Affinity_wt_parsed",
    "kon_mut (M^(-1)s^(-1))", "kon_mut_parsed", "kon_wt (M^(-1)s^(-1))", "kon_wt_parsed",
    "koff_mut (s^(-1))", "koff_mut_parsed", "koff_wt (s^(-1))", "koff_wt_parsed",
    "dH_mut (kcal mol^(-1))", "dH_wt (kcal mol^(-1))",
    "dS_mut (cal mol^(-1) K^(-1))", "dS_wt (cal mol^(-1) K^(-1))",
    "Temperature",
    "DDG", "ddg_foldx", "ddg_foldx_calibrated", "prediction_error",
    "secondary_structure",
    "resnum", "resolution", "msf",
    # prot_ columns are intentionally NOT dropped — they are protein-level features
}

# Classification thresholds (kcal/mol)
PREDICTION_ERROR_THRESHOLDS = (0.5, 1.5)   # accurate / moderate / large_error
DDG_THRESHOLDS              = (-0.5, 0.5)  # stabilising / neutral / destabilising
ERROR_LABELS  = ["accurate", "moderate_error", "large_error"]
CLASS_LABELS  = ["stabilising", "neutral", "destabilising"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all per-structure CSVs (4-char PDB ID filenames only)."""
    csvs = [f for f in results_dir.glob("*.csv")
            if re.match(r"^[A-Z0-9]{4}\.csv$", f.name)]
    if not csvs:
        log.error("No per-structure CSV files found in %s", results_dir.resolve())
        sys.exit(1)
    dfs = [pd.read_csv(f) for f in csvs]
    for df, f in zip(dfs, csvs):
        log.info("Loaded %s (%d rows)", f.name, len(df))
    combined = pd.concat(dfs, ignore_index=True)
    log.info("Total: %d mutations across %d structures",
             len(combined), combined["pdb_id"].nunique())
    return combined

# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def prepare_xy(df: pd.DataFrame, target: str = DEFAULT_TARGET,
               include_foldx: bool = False,
               pdb_paths: dict = None) -> tuple:
    """
    Build feature matrix X and target y.
    Returns (X, y, groups, feature_names, df_valid).
    df_valid retains all columns (including target and resnum) for downstream use.
    """
    df = build_features(df, pdb_paths=pdb_paths)

    if target not in df.columns:
        log.error("Target '%s' not found. Available: %s",
                  target, [c for c in ["prediction_error", "DDG"] if c in df.columns])
        sys.exit(1)

    n_notnull = df[target].notna().sum()
    log.info("Target '%s': %d/%d non-null values", target, n_notnull, len(df))
    if n_notnull == 0:
        log.error("Target '%s' is all NaN — FoldX may not have run, "
                  "or try --target DDG.", target)
        sys.exit(1)

    drop = set(ALWAYS_DROP)
    drop.discard(target)
    if include_foldx and target == "DDG" and "ddg_foldx" in df.columns:
        drop.discard("ddg_foldx")
        log.info("Including ddg_foldx as a feature")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in drop and c != target]

    if not feature_cols:
        log.error("No feature columns found after filtering")
        sys.exit(1)

    valid  = df.dropna(subset=feature_cols + [target]).copy()
    X      = valid[feature_cols]
    y      = valid[target]
    groups = valid["pdb_id"]

    log.info("Target: '%s' | %d features | %d rows | %d structures",
             target, len(feature_cols), len(X), groups.nunique())
    log.info("Features: %s", feature_cols)
    return X, y, groups, feature_cols, valid

# ---------------------------------------------------------------------------
# PDB path building
# ---------------------------------------------------------------------------

def build_pdb_paths(df: pd.DataFrame) -> dict:
    """Build {pdb_id: Path} for all structures, warn on missing files."""
    from config import STRUCTURES_DIR
    pdb_paths = {
        pdb_id: _PROJECT_ROOT / "data" / "structures" / f"{pdb_id.lower()}.pdb"
        for pdb_id in df["pdb_id"].unique()
    }
    found   = [p for p, path in pdb_paths.items() if path.exists()]
    missing = [p for p, path in pdb_paths.items() if not path.exists()]
    log.info("PDB files found: %s", found)
    if missing:
        log.warning("PDB files not found for %s — structural features skipped", missing)
    return {p: path for p, path in pdb_paths.items() if path.exists()}

# ---------------------------------------------------------------------------
# Shared CV: Leave-One-Protein-Out
# ---------------------------------------------------------------------------

def lopo_cv(model, X: pd.DataFrame, y: pd.Series,
            groups: pd.Series, model_name: str,
            scorer, score_name: str) -> pd.DataFrame:
    """
    Generic Leave-One-Protein-Out CV.
    scorer(y_true, y_pred) -> float
    score_name: label for the metric column in output DataFrame.
    """
    proteins = groups.unique()

    if len(proteins) < 2:
        log.warning("%s: only 1 structure — LOPO requires at least 2.", model_name)
        return pd.DataFrame()

    records = []
    for held_out in proteins:
        train_mask = groups != held_out
        test_mask  = groups == held_out
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 5 or len(X_test) < 2:
            log.warning("Skipping fold %s — too few samples", held_out)
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scorer(y_test, y_pred)

        # RMSE only makes sense for numeric targets
        try:
            rmse = float(np.sqrt(np.mean((np.array(y_test, dtype=float) -
                                          np.array(y_pred, dtype=float)) ** 2)))
        except (ValueError, TypeError):
            rmse = float("nan")

        log.debug("%s | held-out: %-8s | n=%3d | %s=%.3f",
                  model_name, held_out, len(X_test), score_name, score)
        records.append({"fold": held_out, "n_test": len(X_test),
                        score_name: score, "rmse": rmse})

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        log.info("%s LOPO — mean %s: %.3f ± %.3f",
                 model_name, score_name,
                 results_df[score_name].mean(), results_df[score_name].std())
    return results_df

# ---------------------------------------------------------------------------
# Shared CV: per-structure GroupKFold by residue position
# ---------------------------------------------------------------------------

def per_structure_cv(model, X: pd.DataFrame, y: pd.Series,
                     groups: pd.Series, model_name: str,
                     pos_groups: pd.Series = None,
                     scorer=None, score_name: str = "score",
                     n_splits: int = 5) -> pd.DataFrame:
    """
    Generic per-structure CV grouped by residue position to prevent leakage.
    Mutations at the same resnum are always in the same fold.
    scorer(y_true, y_pred) -> float
    """
    proteins = groups.unique()
    records  = []

    for prot in proteins:
        mask = groups == prot
        X_p  = X[mask]
        y_p  = y[mask]
        pg   = pos_groups[mask].values if pos_groups is not None else np.arange(len(X_p))

        n_unique_pos  = len(np.unique(pg))
        actual_splits = min(n_splits, n_unique_pos)

        if n_unique_pos < 2:
            log.warning("Skipping %s — too few unique positions (%d)", prot, n_unique_pos)
            continue

        scores, rmses = [], []
        for train_idx, test_idx in GroupKFold(n_splits=actual_splits).split(X_p, y_p, groups=pg):
            X_train, X_test = X_p.iloc[train_idx], X_p.iloc[test_idx]
            y_train, y_test = y_p.iloc[train_idx], y_p.iloc[test_idx]
            if len(X_train) < 2 or len(X_test) < 1:
                continue
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(scorer(y_test, y_pred))
            try:
                rmses.append(float(np.sqrt(np.mean(
                    (np.array(y_test, dtype=float) - np.array(y_pred, dtype=float)) ** 2))))
            except (ValueError, TypeError):
                rmses.append(float("nan"))

        if not scores:
            continue

        sc = np.array(scores)
        rm = np.array(rmses)
        log.debug("%s | structure: %-8s | n=%3d | pos=%3d | %s=%.3f ± %.3f",
                  model_name, prot, len(X_p), n_unique_pos,
                  score_name, sc.mean(), sc.std())
        records.append({
            "structure":          prot,
            "n":                  len(X_p),
            "n_positions":        n_unique_pos,
            f"{score_name}_mean": sc.mean(),
            f"{score_name}_std":  sc.std(),
            "rmse_mean":          rm.mean(),
            "rmse_std":           rm.std(),
        })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Shared: feature importances
# ---------------------------------------------------------------------------

def get_feature_importances(rf_model, X, y, feature_names) -> pd.DataFrame:
    """Fit RF on full data and return impurity + permutation importances.
    Automatically detects regression vs classification from target dtype."""
    rf_model.fit(X, y)
    # Find the RF step regardless of pipeline naming
    rf_step = None
    for step in rf_model.named_steps.values():
        if hasattr(step, "feature_importances_"):
            rf_step = step
            break
    if rf_step is None:
        raise ValueError("No estimator with feature_importances_ found in pipeline")
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf_step.feature_importances_,
    })
    # Use appropriate scoring metric based on target type
    is_classification = pd.api.types.is_categorical_dtype(y) or y.dtype == object
    scoring = "balanced_accuracy" if is_classification else "r2"
    perm = permutation_importance(rf_model, X, y, n_repeats=20,
                                  random_state=42, scoring=scoring)
    imp_df["perm_importance"]     = perm.importances_mean
    imp_df["perm_importance_std"] = perm.importances_std
    return imp_df.sort_values("perm_importance", ascending=False)

# ---------------------------------------------------------------------------
# Shared: feature importance plot
# ---------------------------------------------------------------------------

def plot_importances(imp_df: pd.DataFrame, title: str, out_path: Path):
    top = imp_df.head(15)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(top["feature"][::-1], top["importance"][::-1], color="#534AB7")
    axes[0].set_xlabel("Mean decrease in impurity")
    axes[0].set_title("RF feature importance (impurity)")
    axes[1].barh(top["feature"][::-1], top["perm_importance"][::-1],
                 xerr=top["perm_importance_std"][::-1], color="#1D9E75", capsize=3)
    axes[1].set_xlabel("Permutation importance (R²)")
    axes[1].set_title("Permutation importance")
    axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()

# ---------------------------------------------------------------------------
# Shared: per-structure bar chart
# ---------------------------------------------------------------------------

def plot_per_structure_results(results: dict, score_col: str,
                               title: str, out_path: Path):
    """Bar chart of per-structure score for each model."""
    models     = list(results.keys())
    structures = sorted(set(
        s for df in results.values() for s in df["structure"].tolist()
    ))
    if not structures:
        return

    x      = np.arange(len(structures))
    width  = 0.8 / max(len(models), 1)
    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    fig, ax = plt.subplots(figsize=(max(8, len(structures) * 2), 5))
    for i, (mname, df) in enumerate(results.items()):
        mean_col = f"{score_col}_mean"
        std_col  = f"{score_col}_std"
        vals = [df.loc[df["structure"] == s, mean_col].values[0]
                if s in df["structure"].values else np.nan for s in structures]
        errs = [df.loc[df["structure"] == s, std_col].values[0]
                if s in df["structure"].values else 0 for s in structures]
        ax.bar(x + i * width, vals, width, yerr=errs,
               label=mname, color=colors[i % len(colors)], alpha=0.8, capsize=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(structures, rotation=30, ha="right")
    ax.set_ylabel(score_col)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()

# ---------------------------------------------------------------------------
# Shared: binarize target for classification
# ---------------------------------------------------------------------------

def binarize_target(y: pd.Series, thresholds=None, labels=None) -> pd.Series:
    """
    Convert continuous target to 3 ordinal classes using fixed thresholds.
    thresholds=(lo, hi): class[0] if y < lo, class[1] if lo<=y<=hi, class[2] if y > hi
    """
    if labels is None:
        labels = CLASS_LABELS
    if thresholds is None:
        q33, q67 = y.quantile([1/3, 2/3])
        log.info("Class boundaries (quantile): %.3f / %.3f", q33, q67)
        thresholds = (q33, q67)
    else:
        log.info("Class boundaries (fixed): %.3f / %.3f", *thresholds)

    lo, hi = thresholds
    classes = pd.cut(y, bins=[-np.inf, lo, hi, np.inf],
                     labels=labels, right=True)
    log.info("Class distribution:\n%s", classes.value_counts().to_string())
    return classes