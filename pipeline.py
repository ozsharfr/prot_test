"""
pipeline.py — Phase 1: Flexibility vs. DDG Prediction Error.
Runs per-structure analysis and produces individual + combined results.
"""

import logging
import pandas as pd

from config import RESULTS_DIR, FIGURES_DIR, SKEMPI_CSV, RESOLUTION_CUTOFF, PILOT_PDB_IDS, INTERFACE_CUTOFF
from skempi import load_skempi, filter_by_resolution
from structures import fetch_structures, fetch_resolutions
from flexibility import assign_flexibility_to_mutations
from analysis import run_statistics, plot_results, calibrate_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Set True to skip FoldX and use |DDG_experimental| as target directly
SKIP_FOLDX = False


def process_structure(pdb_id: str, df: pd.DataFrame, pdb_paths: dict) -> tuple[pd.DataFrame, dict]:
    """
    Run the full pipeline for a single complex:
      - ANM flexibility scoring
      - FoldX DDG prediction (or |DDG_exp| if SKIP_FOLDX)
      - Statistics and per-structure figure

    Returns (enriched_df, stats_dict)
    """
    log.info("=" * 50)
    log.info("Processing %s (%d mutations)", pdb_id, len(df))

    # ANM flexibility
    df = assign_flexibility_to_mutations(df, pdb_paths)
    df_if = df[df["is_interface"] == True].copy()
    log.info("%s: %d interface mutations", pdb_id, len(df_if))

    if df_if.empty:
        log.warning("%s: no interface mutations — skipping", pdb_id)
        return df_if, {}

    # DDG prediction
    if SKIP_FOLDX:
        log.info("%s: SKIP_FOLDX=True — using |DDG_experimental|", pdb_id)
        df_if["prediction_error"] = df_if["DDG"].abs()
    else:
        from foldx import run_foldx_for_group
        log.info("%s: running FoldX...", pdb_id)
        ddg_list = run_foldx_for_group(pdb_id, df_if, pdb_paths)
        df_if["ddg_foldx"] = ddg_list
        df_if = calibrate_predictions(df_if)

    # Per-structure stats and figure
    results = run_statistics(df_if, label=pdb_id)
    plot_results(df_if, label=pdb_id, suffix=f"_{pdb_id}")

    return df_if, results


def print_summary(all_results: dict):
    """Print a summary table across all structures."""
    print(f"\n{'='*65}")
    print(f"{'PDB':<8} {'N':>5} {'Spearman ρ':>11} {'p-value':>9} {'MW p':>9}")
    print(f"{'-'*65}")
    for pdb_id, r in all_results.items():
        if not r:
            print(f"{pdb_id:<8}  -- skipped (no interface mutations)")
            continue
        rho = f"{r['spearman_rho']:.3f}" if not pd.isna(r['spearman_rho']) else "  nan"
        p   = f"{r['spearman_p']:.4f}"   if not pd.isna(r['spearman_p'])   else "   nan"
        mwp = f"{r['mw_p']:.4f}"         if r['mw_p'] is not None and not pd.isna(r['mw_p']) else "   nan"
        print(f"{pdb_id:<8} {r['n']:>5} {rho:>11} {p:>9} {mwp:>9}")
    print(f"{'='*65}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load SKEMPI
    skempi = load_skempi(SKEMPI_CSV)

    # 2. Restrict to pilot set
    if PILOT_PDB_IDS is not None:
        skempi = skempi[skempi["pdb_id"].isin(PILOT_PDB_IDS)].copy()
        log.info("Pilot mode: %d complexes → %d mutations",
                 len(PILOT_PDB_IDS), len(skempi))

    # 3. Fetch structures
    pdb_paths = fetch_structures(skempi["pdb_id"].unique().tolist())

    # 4. Resolution filter (soft — warns but doesn't crash if API fails)
    resolutions = fetch_resolutions(list(pdb_paths.keys()))
    log.info("Resolutions fetched: %s", resolutions)
    if resolutions:
        skempi = filter_by_resolution(skempi, resolutions, RESOLUTION_CUTOFF)
    else:
        log.warning("No resolutions returned — skipping resolution filter.")
    log.info("Mutations after resolution filter: %d", len(skempi))

    # 5. Per-structure loop
    all_results: dict = {}
    all_dfs: list = []

    for pdb_id, group in skempi.groupby("pdb_id"):
        df_out, results = process_structure(pdb_id, group.copy(), pdb_paths)
        all_results[pdb_id] = results
        if not df_out.empty:
            all_dfs.append(df_out)

    # 6. Combined analysis across all structures
    if len(all_dfs) > 1:
        log.info("=" * 50)
        log.info("Running combined analysis across %d structures", len(all_dfs))
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_results = run_statistics(combined, label="COMBINED")
        plot_results(combined, label="All structures", suffix="_combined")
        all_results["COMBINED"] = combined_results

        combined_csv = RESULTS_DIR / "mutations_combined.csv"
        combined.to_csv(combined_csv, index=False)
        log.info("Combined results saved to %s", combined_csv)

    elif all_dfs:
        all_dfs[0].to_csv(RESULTS_DIR / "mutations_combined.csv", index=False)

    # 7. Summary table
    print_summary(all_results)


if __name__ == "__main__":
    main()