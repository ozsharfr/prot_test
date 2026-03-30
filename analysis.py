"""
analysis.py — Calibration, error computation, statistics, and plotting.
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
    slope, intercept, r, p, _ = stats.linregress(valid["ddg_foldx"], valid["DDG"])
    log.info("FoldX global calibration: r=%.3f, p=%.4f, slope=%.3f", r, p, slope)

    df = df.copy()
    df["ddg_foldx_calibrated"] = slope * df["ddg_foldx"] + intercept
    df["prediction_error"]     = (df["ddg_foldx_calibrated"] - df["DDG"]).abs()
    return df


def run_statistics(df: pd.DataFrame) -> dict:
    """
    Test whether flexibility (msf_z) predicts prediction error.

    Tests:
      - Spearman correlation across all mutations
      - Mann-Whitney U between low and high flexibility tertiles

    Returns a dict of results.
    """
    valid = df.dropna(subset=["msf_z", "prediction_error"])

    rho, p_spearman = stats.spearmanr(valid["msf_z"], valid["prediction_error"])
    log.info("Spearman rho=%.3f, p=%.4f (n=%d)", rho, p_spearman, len(valid))

    tertiles = pd.qcut(valid["msf_z"], q=3, labels=["low", "mid", "high"])
    low_err  = valid.loc[tertiles == "low",  "prediction_error"]
    high_err = valid.loc[tertiles == "high", "prediction_error"]
    u_stat, p_mw = stats.mannwhitneyu(low_err, high_err, alternative="less")
    log.info("Mann-Whitney U=%.1f, p=%.4f (low vs high flex)", u_stat, p_mw)

    return {
        "spearman_rho": rho,
        "spearman_p":   p_spearman,
        "mw_u":         u_stat,
        "mw_p":         p_mw,
        "n":            len(valid),
        "n_low":        len(low_err),
        "n_high":       len(high_err),
    }


def plot_results(df: pd.DataFrame):
    """
    Generate and save two summary figures:
      1. Scatter: MSF z-score vs. prediction error with regression line
      2. Boxplot: prediction error split by flexibility tertile
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    valid = df.dropna(subset=["msf_z", "prediction_error"]).copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Scatter -----------------------------------------------------------
    ax = axes[0]
    ax.scatter(valid["msf_z"], valid["prediction_error"],
               alpha=0.3, s=12, color="#534AB7")
    m, b, *_ = stats.linregress(valid["msf_z"], valid["prediction_error"])
    xs = np.linspace(valid["msf_z"].min(), valid["msf_z"].max(), 100)
    ax.plot(xs, m * xs + b, color="#D85A30", linewidth=1.5)
    ax.set_xlabel("MSF z-score (flexibility)")
    ax.set_ylabel("|ΔΔG predicted − experimental| (kcal/mol)")
    ax.set_title("Flexibility vs. prediction error")

    # — Boxplot -----------------------------------------------------------
    ax = axes[1]
    valid["tertile"] = pd.qcut(
        valid["msf_z"], q=3, labels=["Low flex", "Mid flex", "High flex"]
    )
    sns.boxplot(data=valid, x="tertile", y="prediction_error", ax=ax,
                palette=["#9FE1CB", "#5DCAA5", "#0F6E56"])
    ax.set_xlabel("Flexibility tertile")
    ax.set_ylabel("|ΔΔG error| (kcal/mol)")
    ax.set_title("Prediction error by flexibility group")

    plt.tight_layout()
    out = FIGURES_DIR / "flexibility_vs_error.png"
    plt.savefig(out, dpi=150)
    log.info("Saved figure: %s", out)
    plt.close()
