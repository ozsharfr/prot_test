"""
analysis.py — Calibration, error computation, statistics, and plotting.
Supports per-structure and combined analysis.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR

log = logging.getLogger(__name__)


def calibrate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a global linear regression of ddg_foldx ~ DDG (experimental),
    then compute per-mutation residuals as the calibrated prediction error.
    Adds columns: ddg_foldx_calibrated, prediction_error
    """
    valid = df.dropna(subset=["ddg_foldx", "DDG"])
    df = df.copy()

    if len(valid) < 3:
        log.warning("Too few FoldX predictions to calibrate (%d rows).", len(valid))
        df["ddg_foldx_calibrated"] = np.nan
        df["prediction_error"]     = np.nan
        return df

    x = valid["ddg_foldx"].to_numpy(dtype=float)
    y = valid["DDG"].to_numpy(dtype=float)

    if np.all(x == x[0]):
        log.warning("All FoldX predictions identical — cannot calibrate.")
        df["ddg_foldx_calibrated"] = np.nan
        df["prediction_error"]     = np.nan
        return df

    result = stats.linregress(x, y)
    slope, intercept, r, p = result.slope, result.intercept, result.rvalue, result.pvalue
    log.info("FoldX calibration: r=%.3f, p=%.4f, slope=%.3f", r, p, slope)

    df["ddg_foldx_calibrated"] = slope * df["ddg_foldx"] + intercept
    df["prediction_error"]     = (df["ddg_foldx_calibrated"] - df["DDG"]).abs()
    return df


def run_statistics(df: pd.DataFrame, label: str = "", flex_col: str = "msf_z") -> dict:
    """
    Spearman correlation and Mann-Whitney U between msf_z and prediction_error.
    label is used only for logging.
    """
    prefix = f"[{label}] " if label else ""
    valid = df.dropna(subset=[flex_col, "prediction_error"])
    null = {"spearman_rho": np.nan, "spearman_p": np.nan,
            "mw_u": np.nan, "mw_p": np.nan,
            "n": len(valid), "n_low": 0, "n_high": 0}

    if len(valid) < 6:
        log.warning("%sToo few rows for statistics (%d)", prefix, len(valid))
        return null

    rho, p_spearman = stats.spearmanr(
        valid[flex_col].to_numpy(dtype=float),
        valid["prediction_error"].to_numpy(dtype=float)
    )
    log.info("%sSpearman rho=%.3f, p=%.4f (n=%d)", prefix, rho, p_spearman, len(valid))

    _, bins = pd.qcut(valid[flex_col], q=3, retbins=True, duplicates="drop")
    n_bins = len(bins) - 1
    if n_bins < 2:
        log.warning("%sCannot split into tertiles (too few unique values)", prefix)
        return {**null, "spearman_rho": rho, "spearman_p": p_spearman, "n": len(valid)}
    all_labels = ["low", "mid", "high"][:n_bins]
    tertiles = pd.cut(valid[flex_col], bins=bins, labels=all_labels,
                      include_lowest=True, duplicates="drop")
    low_err  = valid.loc[tertiles == "low",  "prediction_error"]
    high_err = valid.loc[tertiles == all_labels[-1], "prediction_error"]

    if len(low_err) < 2 or len(high_err) < 2:
        log.warning("%sTertile groups too small (low=%d, high=%d)",
                    prefix, len(low_err), len(high_err))
        return {**null, "spearman_rho": rho, "spearman_p": p_spearman, "n": len(valid)}

    u_stat, p_mw = stats.mannwhitneyu(low_err, high_err, alternative="less")
    log.info("%sMann-Whitney U=%.1f, p=%.4f", prefix, u_stat, p_mw)

    return {
        "spearman_rho": rho,    "spearman_p": p_spearman,
        "mw_u":         u_stat, "mw_p":        p_mw,
        "n":            len(valid),
        "n_low":        len(low_err),
        "n_high":       len(high_err),
    }


# Flexibility score columns to analyse — residue + neighborhood windows
FLEX_SCORES = {
    "msf_z":              "Residue MSF z",
    "msf_z_neighbors_2":  "±2 neighbor MSF z",
    "msf_z_neighbors_4":  "±4 neighbor MSF z",
}


def plot_results(df: pd.DataFrame, label: str = "", suffix: str = ""):
    """
    3-column figure — one column per flexibility score (residue, ±2, ±4 neighbors).
    Each column: scatter (flex vs error) on top, boxplot by tertile on bottom.
    Saves to FIGURES_DIR/flexibility_vs_error{suffix}.png
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    scores = {k: v for k, v in FLEX_SCORES.items() if k in df.columns}
    n_scores = len(scores)
    title_prefix = f"{label}: " if label else ""

    fig, axes = plt.subplots(2, n_scores, figsize=(6 * n_scores, 10))
    if n_scores == 1:
        axes = axes.reshape(2, 1)

    for col_i, (flex_col, flex_label) in enumerate(scores.items()):
        valid = df.dropna(subset=[flex_col, "prediction_error"]).copy()

        if len(valid) < 6:
            log.warning("[%s] Not enough data for %s (%d rows)", label, flex_col, len(valid))
            continue

        rho, p = stats.spearmanr(
            valid[flex_col].to_numpy(dtype=float),
            valid["prediction_error"].to_numpy(dtype=float)
        )

        # — Scatter -------------------------------------------------------
        ax = axes[0, col_i]
        ax.scatter(valid[flex_col], valid["prediction_error"],
                   alpha=0.4, s=18, color="#534AB7")
        _lr = stats.linregress(
            valid[flex_col].to_numpy(dtype=float),
            valid["prediction_error"].to_numpy(dtype=float)
        )
        xs = np.linspace(valid[flex_col].min(), valid[flex_col].max(), 100)
        ax.plot(xs, _lr.slope * xs + _lr.intercept, color="#D85A30", linewidth=1.5)
        ax.set_xlabel(flex_label)
        ax.set_ylabel("Prediction error (kcal/mol)")
        ax.set_title(f"{title_prefix}{flex_label}\nρ={rho:.3f}, p={p:.3f}, n={len(valid)}")

        # — Boxplot -------------------------------------------------------
        ax = axes[1, col_i]
        _, bins = pd.qcut(valid[flex_col], q=3, retbins=True, duplicates="drop")
        n_bins = len(bins) - 1
        all_labels = ["Low", "Mid", "High"][:n_bins]
        valid["tertile"] = pd.cut(valid[flex_col], bins=bins,
                                   labels=all_labels, include_lowest=True,
                                   duplicates="drop")
        sns.boxplot(data=valid, x="tertile", y="prediction_error",
                    hue="tertile", legend=False, ax=ax,
                    palette=["#9FE1CB", "#5DCAA5", "#0F6E56"])
        ax.set_xlabel(f"{flex_label} tertile")
        ax.set_ylabel("Prediction error (kcal/mol)")
        ax.set_title(f"Error by {flex_label} group")

    plt.suptitle(f"{title_prefix}Flexibility vs. Prediction Error", fontsize=13, y=1.01)
    plt.tight_layout()
    fname = f"flexibility_vs_error{suffix}.png"
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Saved figure: %s", out)
    plt.close()