"""
pipeline.py — Phase 1: Flexibility vs. DDG Prediction Error.
Runs per-structure analysis and produces individual + combined results.

Resume behaviour: if results/<PDB_ID>.csv already exists, that structure
is skipped and loaded from disk instead of recomputed.
"""

import logging
import pandas as pd

from config import RESULTS_DIR, FIGURES_DIR, SKEMPI_CSV, RESOLUTION_CUTOFF, PILOT_PDB_IDS, INTERFACE_CUTOFF
from skempi import load_skempi, filter_by_resolution
from structures import fetch_structures, fetch_resolutions
from flexibility import assign_flexibility_to_mutations
from analysis import run_statistics, plot_results, calibrate_predictions, FLEX_SCORES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SKIP_FOLDX = False

SCORE_LABELS = {
    "msf_z":             "Residue",
    "msf_z_neighbors_2": "±2 neighbors",
    "msf_z_neighbors_4": "±4 neighbors",
}


def per_structure_csv(pdb_id: str) -> "Path":
    from config import RESULTS_DIR
    return RESULTS_DIR / f"{pdb_id}.csv"


def compute_statistics(df_if: pd.DataFrame, pdb_id: str) -> dict:
    """Run statistics for all flexibility scores on a processed dataframe."""
    results = {}
    for flex_col in FLEX_SCORES:
        if flex_col in df_if.columns:
            results[flex_col] = run_statistics(
                df_if,
                label=f"{pdb_id}/{SCORE_LABELS.get(flex_col, flex_col)}",
                flex_col=flex_col
            )
    return results


def process_structure(pdb_id: str, df: pd.DataFrame, pdb_paths: dict) -> tuple[pd.DataFrame, dict]:
    """
    Run the full pipeline for a single complex.
    Saves result to results/<PDB_ID>.csv on completion.
    If that file already exists, loads it and skips recomputation.

    Returns (enriched_df, {flex_col: stats_dict})
    """
    out_csv = per_structure_csv(pdb_id)

    # --- Resume: load from disk if already completed ---
    if out_csv.exists():
        log.info("RESUME: %s already done — loading from %s", pdb_id, out_csv)
        df_if = pd.read_csv(out_csv)
        results = compute_statistics(df_if, pdb_id)
        plot_results(df_if, label=pdb_id, suffix=f"_{pdb_id}")
        return df_if, results

    # --- Fresh run ---
    log.info("=" * 50)
    log.info("Processing %s (%d mutations)", pdb_id, len(df))

    df = assign_flexibility_to_mutations(df, pdb_paths)
    df_if = df[df["is_interface"] == True].copy()
    log.info("%s: %d interface mutations", pdb_id, len(df_if))

    if df_if.empty:
        log.warning("%s: no interface mutations — skipping", pdb_id)
        return df_if, {}

    if SKIP_FOLDX:
        log.info("%s: SKIP_FOLDX=True — using |DDG_experimental|", pdb_id)
        df_if["prediction_error"] = df_if["DDG"].abs()
    else:
        from foldx import run_foldx_for_group
        log.info("%s: running FoldX...", pdb_id)
        df_if["ddg_foldx"] = run_foldx_for_group(pdb_id, df_if, pdb_paths)
        df_if = calibrate_predictions(df_if)

    # Save per-structure result before stats/plotting
    # so even if plotting crashes, the data is safe
    df_if.to_csv(out_csv, index=False)
    log.info("%s: saved to %s", pdb_id, out_csv)

    results = compute_statistics(df_if, pdb_id)
    plot_results(df_if, label=pdb_id, suffix=f"_{pdb_id}")
    return df_if, results


def print_summary(all_results: dict):
    """Print a summary table — one row per structure per flexibility score."""
    w = 78
    print(f"\n{'='*w}")
    print(f"{'PDB':<10} {'Score':<16} {'N':>5} {'Spearman ρ':>11} {'p-value':>9} {'MW p':>9}")
    print(f"{'-'*w}")
    for pdb_id, results in all_results.items():
        if not results:
            print(f"{pdb_id:<10}  -- skipped (no interface mutations)")
            continue
        for flex_col, r in results.items():
            slabel = SCORE_LABELS.get(flex_col, flex_col)
            if not r or r["n"] == 0:
                print(f"{pdb_id:<10} {slabel:<16}  -- insufficient data")
                continue
            def fmt(v, fmt_str):
                return f"{v:{fmt_str}}" if v is not None and not pd.isna(v) else "    nan"
            print(f"{pdb_id:<10} {slabel:<16} {r['n']:>5}"
                  f" {fmt(r['spearman_rho'], '>11.3f')}"
                  f" {fmt(r['spearman_p'],   '>9.4f')}"
                  f" {fmt(r['mw_p'],         '>9.4f')}")
        print(f"{'-'*w}")
    print(f"{'='*w}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    skempi = load_skempi(SKEMPI_CSV)

    if PILOT_PDB_IDS is not None:
        skempi = skempi[skempi["pdb_id"].isin(PILOT_PDB_IDS)].copy()
        log.info("Pilot mode: %d complexes → %d mutations",
                 len(PILOT_PDB_IDS), len(skempi))

    pdb_paths = fetch_structures(skempi["pdb_id"].unique().tolist())

    resolutions = fetch_resolutions(list(pdb_paths.keys()))
    log.info("Resolutions fetched: %s", resolutions)
    if resolutions:
        skempi = filter_by_resolution(skempi, resolutions, RESOLUTION_CUTOFF)
    else:
        log.warning("No resolutions returned — skipping resolution filter.")
    log.info("Mutations after resolution filter: %d", len(skempi))

    all_results: dict = {}
    all_dfs: list = []

    for pdb_id, group in skempi.groupby("pdb_id"):
        try:
            df_out, results = process_structure(pdb_id, group.copy(), pdb_paths)
            all_results[pdb_id] = results
            if not df_out.empty:
                all_dfs.append(df_out)
        except Exception as e:
            import traceback
            log.error("FAILED %s: %s", pdb_id, e)
            traceback.print_exc()
            log.error("Continuing with next structure...")
            all_results[pdb_id] = {}

    # Combined analysis
    if len(all_dfs) > 1:
        log.info("=" * 50)
        log.info("Running combined analysis across %d structures", len(all_dfs))
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_results = {}
        for flex_col in FLEX_SCORES:
            if flex_col in combined.columns:
                combined_results[flex_col] = run_statistics(
                    combined,
                    label=f"COMBINED/{SCORE_LABELS.get(flex_col, flex_col)}",
                    flex_col=flex_col
                )
        plot_results(combined, label="All structures", suffix="_combined")
        all_results["COMBINED"] = combined_results
        combined.to_csv(RESULTS_DIR / "mutations_combined.csv", index=False)
        log.info("Combined results saved.")
    elif all_dfs:
        all_dfs[0].to_csv(RESULTS_DIR / "mutations_combined.csv", index=False)

    print_summary(all_results)


if __name__ == "__main__":
    main()