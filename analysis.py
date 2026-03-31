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


def run_statistics(df: pd.DataFrame, label: str = "") -> dict:
    """
    Spearman correlation and Mann-Whitney U between msf_z and prediction_error.
    label is used only for logging.
    """
    prefix = f"[{label}] " if label else ""
    valid = df.dropna(subset=["msf_z", "prediction_error"])
    null = {"spearman_rho": np.nan, "spearman_p": np.nan,
            "mw_u": np.nan, "mw_p": np.nan,
            "n": len(valid), "n_low": 0, "n_high": 0}

    if len(valid) < 6:
        log.warning("%sToo few rows for statistics (%d)", prefix, len(valid))
        return null

    rho, p_spearman = stats.spearmanr(
        valid["msf_z"].to_numpy(dtype=float),
        valid["prediction_error"].to_numpy(dtype=float)
    )
    log.info("%sSpearman rho=%.3f, p=%.4f (n=%d)", prefix, rho, p_spearman, len(valid))

    tertiles = pd.qcut(valid["msf_z"], q=3, labels=["low", "mid", "high"],
                       duplicates="drop")
    low_err  = valid.loc[tertiles == "low",  "prediction_error"]
    high_err = valid.loc[tertiles == "high", "prediction_error"]

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


def plot_results(df: pd.DataFrame, label: str = "", suffix: str = ""):
    """
    Two-panel figure: scatter (msf_z vs prediction_error) + boxplot by tertile.
    Saves to FIGURES_DIR/flexibility_vs_error{suffix}.png
    label is shown in plot titles.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    valid = df.dropna(subset=["msf_z", "prediction_error"]).copy()

    if len(valid) < 6:
        log.warning("[%s] Not enough data to plot (%d rows) — skipping.", label, len(valid))
        return

    title_prefix = f"{label}: " if label else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Scatter -----------------------------------------------------------
    ax = axes[0]
    ax.scatter(valid["msf_z"], valid["prediction_error"],
               alpha=0.4, s=18, color="#534AB7")
    _lr = stats.linregress(
        valid["msf_z"].to_numpy(dtype=float),
        valid["prediction_error"].to_numpy(dtype=float)
    )
    xs = np.linspace(valid["msf_z"].min(), valid["msf_z"].max(), 100)
    ax.plot(xs, _lr.slope * xs + _lr.intercept, color="#D85A30", linewidth=1.5)
    rho, p = stats.spearmanr(valid["msf_z"], valid["prediction_error"])
    ax.set_xlabel("MSF z-score (flexibility)")
    ax.set_ylabel("Prediction error (kcal/mol)")
    ax.set_title(f"{title_prefix}Flexibility vs. error\nSpearman ρ={rho:.3f}, p={p:.3f}")

    # — Boxplot -----------------------------------------------------------
    ax = axes[1]
    valid["tertile"] = pd.qcut(
        valid["msf_z"], q=3,
        labels=["Low flex", "Mid flex", "High flex"],
        duplicates="drop"
    )
    sns.boxplot(data=valid, x="tertile", y="prediction_error",
                hue="tertile", legend=False, ax=ax,
                palette=["#9FE1CB", "#5DCAA5", "#0F6E56"])
    ax.set_xlabel("Flexibility tertile")
    ax.set_ylabel("Prediction error (kcal/mol)")
    ax.set_title(f"{title_prefix}Error by flexibility group\n(n={len(valid)})")

    plt.tight_layout()
    fname = f"flexibility_vs_error{suffix}.png"
    out = FIGURES_DIR / fname
    plt.savefig(out, dpi=150)
    log.info("Saved figure: %s", out)
    plt.close()