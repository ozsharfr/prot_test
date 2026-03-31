"""
regressor.py — Predict DDG target from biophysical features.

Validation strategy: Leave-One-Protein-Out (LOPO) CV.
Train on all structures except one, test on the held-out structure.
This tests whether the model generalises to unseen proteins.
"""

import sys
import os
import logging
from pathlib import Path

# --- Path setup: must come before any local imports ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — avoids tkinter threading issues on Windows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder

from features import build_features
from config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DEFAULT_TARGET = "prediction_error"

# Class boundaries (kcal/mol) — biophysically motivated.
#
# prediction_error thresholds:
#   < 0.5  = FoldX accurate (within experimental noise ~0.3–0.5 kcal/mol)
#   0.5–1.5 = moderate error
#   > 1.5  = large error (FoldX fails badly)
#
# DDG thresholds:
#   < -0.5 = stabilising mutation
#   -0.5 to +0.5 = neutral (within noise floor)
#   > +0.5 = destabilising
PREDICTION_ERROR_THRESHOLDS = (0.5, 1.5)
DDG_THRESHOLDS              = (-0.5, 0.5)
ERROR_LABELS = ["accurate", "moderate_error", "large_error"]
CLASS_LABELS = ["stabilising", "neutral", "destabilising"]

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
    # these encode protein identity or are raw/unnormalized versions
    "resnum", "resolution", "msf",
    # prot_ columns are intentionally NOT dropped — they are protein-level features
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> pd.DataFrame:
    import re
    csvs = [f for f in results_dir.glob("*.csv")
            if re.match(r"^[A-Z0-9]{4}\.csv$", f.name)]  # only PDB ID named files
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
    df = build_features(df, pdb_paths=pdb_paths)
    if target not in df.columns:
        log.error("Target '%s' not found. Available: %s",
                  target, [c for c in ["prediction_error", "DDG"] if c in df.columns])
        sys.exit(1)

    n_total   = len(df)
    n_notnull = df[target].notna().sum()
    log.info("Target '%s': %d/%d non-null values", target, n_notnull, n_total)
    if n_notnull == 0:
        log.error("Target '%s' is all NaN — FoldX may not have run, "
                  "or try --target DDG to use experimental values instead.", target)
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

    # Keep pdb_id alongside for LOPO splitting
    valid = df.dropna(subset=feature_cols + [target]).copy()
    X      = valid[feature_cols]
    y      = valid[target]
    groups = valid["pdb_id"]

    log.info("Target: '%s' | %d features | %d rows | %d structures",
             target, len(feature_cols), len(X), groups.nunique())
    log.info("Features: %s", feature_cols)
    # Return valid (enriched df with all columns) so callers can access target
    return X, y, groups, feature_cols, valid


# ---------------------------------------------------------------------------
# Leave-One-Protein-Out CV
# ---------------------------------------------------------------------------

def lopo_cv(model, X: pd.DataFrame, y: pd.Series,
            groups: pd.Series, model_name: str) -> pd.DataFrame:
    """
    Leave-One-Protein-Out cross-validation.

    For each unique protein, train on all others and evaluate on held-out.
    Returns a DataFrame with per-fold results.
    """
    proteins = groups.unique()

    if len(proteins) < 2:
        log.warning("Only 1 structure — LOPO CV requires at least 2. "
                    "Falling back to 5-fold CV.")
        from sklearn.model_selection import KFold, cross_val_score
        if len(X) < 2:
            log.warning("%s: not enough samples (%d) for CV", model_name, len(X))
            return pd.DataFrame(columns=["fold", "r2", "rmse", "n_test"])
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=42)
        r2   = cross_val_score(model, X, y, cv=cv, scoring="r2")
        rmse = np.sqrt(-cross_val_score(model, X, y, cv=cv,
                                        scoring="neg_mean_squared_error"))
        log.info("%s (5-fold) — R²: %.3f ± %.3f | RMSE: %.3f ± %.3f",
                 model_name, r2.mean(), r2.std(), rmse.mean(), rmse.std())
        return pd.DataFrame({"fold": ["5-fold"], "r2": [r2.mean()],
                             "rmse": [rmse.mean()], "n_test": [len(X)]})

    records = []
    for held_out in proteins:
        train_mask = groups != held_out
        test_mask  = groups == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(X_train) < 5 or len(X_test) < 2:
            log.warning("Skipping fold %s — too few samples (train=%d, test=%d)",
                        held_out, len(X_train), len(X_test))
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        log.info("%s | held-out: %-8s | n_test=%3d | R²=%6.3f | RMSE=%.3f",
                 model_name, held_out, len(X_test), r2, rmse)
        records.append({
            "fold":    held_out,
            "n_test":  len(X_test),
            "r2":      r2,
            "rmse":    rmse,
        })

    results_df = pd.DataFrame(records)
    if not results_df.empty:
        log.info("%s LOPO summary — mean R²: %.3f ± %.3f | mean RMSE: %.3f ± %.3f",
                 model_name,
                 results_df["r2"].mean(),   results_df["r2"].std(),
                 results_df["rmse"].mean(), results_df["rmse"].std())
    return results_df


# ---------------------------------------------------------------------------
# Feature importances (trained on full dataset)
# ---------------------------------------------------------------------------

def get_feature_importances(rf_model, X, y, feature_names):
    """Fit RF on full data — used only for feature ranking, not evaluation."""
    rf_model.fit(X, y)
    imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf_model.named_steps["rf"].feature_importances_,
    })
    perm = permutation_importance(rf_model, X, y, n_repeats=20,
                                  random_state=42, scoring="r2")
    imp_df["perm_importance"]     = perm.importances_mean
    imp_df["perm_importance_std"] = perm.importances_std
    return imp_df.sort_values("perm_importance", ascending=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lopo_results(all_cv: dict, title: str, out_path: Path):
    """Bar chart of per-fold R² for each model, grouped by held-out protein."""
    models   = list(all_cv.keys())
    proteins = sorted(set(
        fold for df in all_cv.values() for fold in df["fold"].tolist()
    ))

    x = np.arange(len(proteins))
    width = 0.8 / len(models)
    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² per fold
    ax = axes[0]
    for i, (mname, df) in enumerate(all_cv.items()):
        r2_vals = [df.loc[df["fold"] == p, "r2"].values[0]
                   if p in df["fold"].values else np.nan
                   for p in proteins]
        ax.bar(x + i * width, r2_vals, width, label=mname, color=colors[i], alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(proteins, rotation=30, ha="right")
    ax.set_ylabel("R² (held-out protein)")
    ax.set_title("LOPO CV — R² per held-out structure")
    ax.legend()

    # Mean R² summary
    ax = axes[1]
    means = [df["r2"].mean() for df in all_cv.values()]
    stds  = [df["r2"].std()  for df in all_cv.values()]
    ax.bar(models, means, yerr=stds, color=colors[:len(models)],
           alpha=0.8, capsize=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean R² across folds")
    ax.set_title("LOPO CV — mean R² by model")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()


def plot_importances(imp_df, title, out_path):
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


def plot_lopo_predictions(model, X, y, groups, target, out_path):
    """
    Scatter of predicted vs actual using LOPO predictions —
    each point is predicted from a model that never saw its protein.
    """
    proteins = groups.unique()
    y_pred_all = pd.Series(index=y.index, dtype=float)

    for held_out in proteins:
        train_mask = groups != held_out
        test_mask  = groups == held_out
        if train_mask.sum() < 5:
            continue
        model.fit(X[train_mask], y[train_mask])
        y_pred_all[test_mask] = model.predict(X[test_mask])

    valid = y_pred_all.dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for prot in proteins:
        mask = groups == prot
        mask = mask & y_pred_all.notna()
        ax.scatter(y[mask], y_pred_all[mask], alpha=0.5, s=20, label=prot)

    lim = [min(y.min(), valid.min()) - 0.1, max(y.max(), valid.max()) + 0.1]
    ax.plot(lim, lim, "k--", linewidth=1)
    r2 = r2_score(y[y_pred_all.notna()], valid)
    ax.set_xlabel(f"Actual {target} (kcal/mol)")
    ax.set_ylabel(f"LOPO predicted {target} (kcal/mol)")
    ax.set_title(f"LOPO predictions (R²={r2:.3f})\nColour = held-out protein")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info("Saved: %s", out_path)
    plt.close()



# ---------------------------------------------------------------------------
# Per-structure CV (KFold within a single protein)
# ---------------------------------------------------------------------------

def per_structure_cv(model, X: pd.DataFrame, y: pd.Series,
                     groups: pd.Series, model_name: str,
                     n_splits: int = 5) -> pd.DataFrame:
    """
    Run KFold CV independently within each structure.
    Useful to see how well the model fits each protein on its own,
    without requiring generalisation across proteins.
    """
    from sklearn.model_selection import KFold

    proteins = groups.unique()
    records = []

    for prot in proteins:
        mask = groups == prot
        X_p, y_p = X[mask], y[mask]

        if len(X_p) < n_splits * 2:
            log.warning("Skipping per-structure CV for %s — too few samples (%d)",
                        prot, len(X_p))
            continue

        cv = KFold(n_splits=min(n_splits, len(X_p)), shuffle=True, random_state=42)
        r2   = cross_val_score(model, X_p, y_p, cv=cv, scoring="r2")
        rmse = np.sqrt(-cross_val_score(model, X_p, y_p, cv=cv,
                                         scoring="neg_mean_squared_error"))

        log.info("%s | structure: %-8s | n=%3d | R²=%6.3f ± %.3f | RMSE=%.3f ± %.3f",
                 model_name, prot, len(X_p),
                 r2.mean(), r2.std(), rmse.mean(), rmse.std())

        records.append({
            "structure":  prot,
            "n":          len(X_p),
            "r2_mean":    r2.mean(),
            "r2_std":     r2.std(),
            "rmse_mean":  rmse.mean(),
            "rmse_std":   rmse.std(),
        })

    return pd.DataFrame(records)


def plot_per_structure_results(per_struct_results: dict, title: str, out_path: Path):
    """Bar chart of per-structure R² for each model."""
    models   = list(per_struct_results.keys())
    # collect all structures across all models
    structures = sorted(set(
        s for df in per_struct_results.values() for s in df["structure"].tolist()
    ))

    if not structures:
        return

    x = np.arange(len(structures))
    width = 0.8 / len(models)
    colors = ["#534AB7", "#1D9E75", "#D85A30"]

    fig, ax = plt.subplots(figsize=(max(8, len(structures) * 2), 5))
    for i, (mname, df) in enumerate(per_struct_results.items()):
        r2_vals = [df.loc[df["structure"] == s, "r2_mean"].values[0]
                   if s in df["structure"].values else np.nan
                   for s in structures]
        err_vals = [df.loc[df["structure"] == s, "r2_std"].values[0]
                    if s in df["structure"].values else 0
                    for s in structures]
        ax.bar(x + i * width, r2_vals, width, yerr=err_vals,
               label=mname, color=colors[i], alpha=0.8, capsize=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(structures, rotation=30, ha="right")
    ax.set_ylabel("R² (within-structure CV)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def binarize_target(y: pd.Series, thresholds=None,
                    labels=None) -> pd.Series:
    """
    Convert continuous target to 3 ordinal classes.

    If thresholds is None, uses quantile-based tertiles.
    If thresholds=(lo, hi), uses fixed boundaries:
        class[0] if y < lo
        class[1] if lo <= y <= hi
        class[2] if y > hi
    """
    if labels is None:
        labels = CLASS_LABELS

    if thresholds is None:
        # Quantile tertiles
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


def lopo_classify(model, X: pd.DataFrame, y_cls: pd.Series,
                  groups: pd.Series, model_name: str) -> pd.DataFrame:
    """
    Leave-One-Protein-Out CV for a classifier.
    Returns per-fold balanced accuracy and full classification report.
    """
    proteins = groups.unique()
    if len(proteins) < 2:
        log.warning("%s: only 1 structure, skipping LOPO classify", model_name)
        return pd.DataFrame()

    records = []
    all_true, all_pred = [], []

    for held_out in proteins:
        train_mask = groups != held_out
        test_mask  = groups == held_out

        X_train, y_train = X[train_mask], y_cls[train_mask]
        X_test,  y_test  = X[test_mask],  y_cls[test_mask]

        if len(X_train) < 5 or len(X_test) < 2:
            continue
        if y_train.nunique() < 2:
            log.warning("%s: only one class in training fold %s — skipping",
                        model_name, held_out)
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        bal_acc = balanced_accuracy_score(y_test, y_pred)
        log.info("%s | held-out: %-8s | n=%3d | balanced_acc=%.3f",
                 model_name, held_out, len(X_test), bal_acc)

        records.append({
            "fold":         held_out,
            "n_test":       len(X_test),
            "balanced_acc": bal_acc,
        })
        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    if all_true:
        log.info("%s LOPO classification report (all folds combined):\n%s",
                 model_name,
                 classification_report(all_true, all_pred, zero_division=0))

    return pd.DataFrame(records)


def per_structure_classify(model, X: pd.DataFrame, y_cls: pd.Series,
                            groups: pd.Series, model_name: str,
                            n_splits: int = 5) -> pd.DataFrame:
    """KFold CV classifier within each structure."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score as cvs
    proteins = groups.unique()
    records = []

    for prot in proteins:
        mask = groups == prot
        X_p, y_p = X[mask], y_cls[mask]

        if len(X_p) < n_splits * 2 or y_p.nunique() < 2:
            log.warning("Skipping classify for %s — too few samples or classes", prot)
            continue

        cv = StratifiedKFold(n_splits=min(n_splits, len(X_p) // 2),
                             shuffle=True, random_state=42)
        scores = cvs(model, X_p, y_p, cv=cv,
                     scoring="balanced_accuracy")

        log.debug("%s | structure: %-8s | n=%3d | bal_acc=%.3f ± %.3f",
                 model_name, prot, len(X_p), scores.mean(), scores.std())

        records.append({
            "structure":       prot,
            "n":               len(X_p),
            "bal_acc_mean":    scores.mean(),
            "bal_acc_std":     scores.std(),
        })

    return pd.DataFrame(records)


def plot_confusion_matrices(model, X: pd.DataFrame, y_cls: pd.Series,
                             groups: pd.Series, labels: list,
                             title: str, out_path: Path):
    """
    One confusion matrix per held-out protein (LOPO predictions).
    """
    proteins = groups.unique()
    n = len(proteins)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, held_out in zip(axes, proteins):
        train_mask = groups != held_out
        test_mask  = groups == held_out
        if train_mask.sum() < 5:
            continue
        model.fit(X[train_mask], y_cls[train_mask])
        y_pred = model.predict(X[test_mask])
        cm = confusion_matrix(y_cls[test_mask], y_pred, labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Held-out: {held_out}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()


def run_classification(df_full: pd.DataFrame, X: pd.DataFrame,
                       groups: pd.Series, feature_names: list,
                       target: str, args):
    """
    Run full classification pipeline for a given target.
    Called from main() for both DDG and prediction_error.
    """
    thresholds = (DDG_THRESHOLDS if target == "DDG"
                  else PREDICTION_ERROR_THRESHOLDS)
    labels = (CLASS_LABELS if target == "DDG" else ERROR_LABELS)

    y_raw = df_full.loc[X.index, target]
    if y_raw.notna().sum() < 6:
        log.warning("Not enough non-null values for classification of '%s'", target)
        return

    y_cls = binarize_target(y_raw, thresholds=thresholds, labels=labels)
    valid  = y_cls.notna()
    X_c, y_c, g_c = X[valid], y_cls[valid], groups[valid]

    clf_models = {
        "RF Classifier": make_pipeline(
            RandomForestClassifier(n_estimators=200, max_depth=5,
                                   class_weight="balanced", random_state=42)
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=42)
        ),
        "GB Classifier": make_pipeline(
            GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                       random_state=42)
        ),
    }

    tag = f"clf_{target}"

    # LOPO
    print(f"\n{'='*65}")
    print(f"Classification — target: {target} | LOPO CV | n={len(X_c)}")
    print(f"{'='*65}")
    lopo_clf_results = {}
    for name, model in clf_models.items():
        cv_df = lopo_classify(model, X_c, y_c, g_c, name)
        lopo_clf_results[name] = cv_df

    print(f"\n{'Model':<25} {'Mean bal_acc':>13} {'Std':>8}")
    print(f"{'-'*50}")
    for name, df in lopo_clf_results.items():
        if df.empty:
            continue
        print(f"{name:<25} {df['balanced_acc'].mean():>13.3f} "
              f"{df['balanced_acc'].std():>8.3f}")

    # Per-structure
    print(f"\n{'='*65}")
    print(f"Per-structure classification — target: {target}")
    print(f"{'='*65}")
    per_struct_clf = {}
    for name, model in clf_models.items():
        ps_df = per_structure_classify(model, X_c, y_c, g_c, name)
        per_struct_clf[name] = ps_df

    print(f"\n{'Structure':<12} {'Model':<25} {'N':>5} {'bal_acc':>9} {'±':>6}")
    print(f"{'-'*60}")
    all_structs = sorted(set(s for df in per_struct_clf.values()
                             for s in df.get("structure", pd.Series()).tolist()))
    for prot in all_structs:
        for name, df in per_struct_clf.items():
            row = df[df["structure"] == prot] if not df.empty else pd.DataFrame()
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"{prot:<12} {name:<25} {int(r['n']):>5} "
                  f"{r['bal_acc_mean']:>9.3f} {r['bal_acc_std']:>6.3f}")
        print()

    # Confusion matrices
    rf_clf = clf_models["RF Classifier"]
    plot_confusion_matrices(
        rf_clf, X_c, y_c, g_c, labels,
        f"RF Classifier confusion matrices — {target}",
        FIGURES_DIR / f"confusion_{tag}.png"
    )

    # Save
    lopo_combined = pd.concat(
        [df.assign(model=n) for n, df in lopo_clf_results.items() if not df.empty],
        ignore_index=True
    )
    lopo_combined.to_csv(RESULTS_DIR / f"lopo_{tag}.csv", index=False)
    ps_combined = pd.concat(
        [df.assign(model=n) for n, df in per_struct_clf.items() if not df.empty],
        ignore_index=True
    )
    ps_combined.to_csv(RESULTS_DIR / f"per_structure_{tag}.csv", index=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        choices=["prediction_error", "DDG"])
    parser.add_argument("--include-foldx", action="store_true",
                        help="Add ddg_foldx as a feature when target=DDG")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results(RESULTS_DIR)

    # Build PDB paths for structural and protein-level features
    from config import STRUCTURES_DIR
    pdb_paths = {
        pdb_id: STRUCTURES_DIR / f"{pdb_id.lower()}.pdb"
        for pdb_id in df["pdb_id"].unique()
    }
    missing = [p for p, path in pdb_paths.items() if not path.exists()]
    found   = [p for p, path in pdb_paths.items() if path.exists()]
    log.info("PDB files found: %s", found)
    if missing:
        log.warning("PDB files not found for %s — structural features will be skipped", missing)
        pdb_paths = {p: path for p, path in pdb_paths.items() if path.exists()}
    log.info("STRUCTURES_DIR resolves to: %s", STRUCTURES_DIR.resolve())

    X, y, groups, feature_names, df_valid = prepare_xy(
        df, target=args.target, include_foldx=args.include_foldx,
        pdb_paths=pdb_paths if pdb_paths else None
    )

    # Save feature matrix for inspection / reuse
    feature_df = X.copy()
    feature_df[args.target] = y.values
    feature_df["pdb_id"]    = groups.values
    out_parquet = RESULTS_DIR / f"feature_matrix_{args.target}.parquet"
    feature_df.to_parquet(out_parquet, index=False)
    log.info("Feature matrix saved to %s (%d rows x %d features)",
             out_parquet, len(feature_df), len(feature_names))

    models = {
        "Random Forest": Pipeline([
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=5,
                                         random_state=42, n_jobs=-1))
        ]),
    }

    print(f"\n{'='*65}")
    print(f"Target: {args.target} | LOPO CV | n={len(X)} | "
          f"structures={groups.nunique()}")
    print(f"{'='*65}")

    all_cv = {}
    for name, model in models.items():
        cv_df = lopo_cv(model, X, y, groups, name)
        all_cv[name] = cv_df

    # Summary table
    print(f"\n{'='*65}")
    print(f"{'Model':<22} {'Mean R²':>9} {'Std R²':>8} {'Mean RMSE':>10}")
    print(f"{'-'*65}")
    for name, cv_df in all_cv.items():
        if cv_df.empty:
            continue
        print(f"{name:<22} {cv_df['r2'].mean():>9.3f} {cv_df['r2'].std():>8.3f} "
              f"{cv_df['rmse'].mean():>10.3f}")
    print(f"{'='*65}")

    # Per-structure CV
    print(f"\n{'='*65}")
    print("Per-structure CV (within each protein)")
    print(f"{'='*65}")
    per_struct_results = {}
    for name, model in models.items():
        ps_df = per_structure_cv(model, X, y, groups, name)
        per_struct_results[name] = ps_df

    # Per-structure summary table
    all_structures = sorted(set(
        s for df in per_struct_results.values()
        for s in df["structure"].tolist()
    ))
    print(f"\n{'Structure':<12} {'Model':<22} {'N':>5} {'R²':>8} {'±':>6} {'RMSE':>8}")
    print(f"{'-'*65}")
    for prot in all_structures:
        for name, df in per_struct_results.items():
            row = df[df["structure"] == prot]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"{prot:<12} {name:<22} {int(r['n']):>5} "
                  f"{r['r2_mean']:>8.3f} {r['r2_std']:>6.3f} {r['rmse_mean']:>8.3f}")
        print()

    plot_per_structure_results(
        per_struct_results,
        f"Per-structure CV — target: {args.target}",
        FIGURES_DIR / f"per_structure_cv_{args.target}.png"
    )

    # Save per-structure results
    ps_combined = pd.concat(
        [df.assign(model=name) for name, df in per_struct_results.items()],
        ignore_index=True
    )
    ps_combined.to_csv(RESULTS_DIR / f"per_structure_cv_{args.target}.csv", index=False)

    # Feature importances (RF, full dataset — for ranking only)
    rf_model = models["Random Forest"]
    imp_df = get_feature_importances(rf_model, X, y, feature_names)
    print(f"\nTop features (permutation importance, full dataset):")
    print(imp_df[["feature", "perm_importance", "perm_importance_std"]]
          .head(10).to_string(index=False))

    # Plots
    plot_lopo_results(all_cv, f"LOPO CV — target: {args.target}",
                      FIGURES_DIR / f"lopo_cv_{args.target}.png")
    plot_importances(imp_df, f"Feature importances — {args.target}",
                     FIGURES_DIR / f"feature_importances_{args.target}.png")
    plot_lopo_predictions(models["Random Forest"], X, y, groups, args.target,
                          FIGURES_DIR / f"lopo_predictions_{args.target}.png")

    imp_df.to_csv(RESULTS_DIR / f"feature_importances_{args.target}.csv", index=False)
    all_cv_combined = pd.concat(
        [df.assign(model=name) for name, df in all_cv.items()],
        ignore_index=True
    )
    all_cv_combined.to_csv(RESULTS_DIR / f"lopo_cv_results_{args.target}.csv", index=False)

    # Classification
    print(f"\n{'='*65}")
    print("Classification analysis")
    print(f"{'='*65}")
    run_classification(df_valid, X, groups, feature_names, args.target, args)

    log.info("Done.")


if __name__ == "__main__":
    main()
