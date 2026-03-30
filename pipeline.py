"""
pipeline.py — Orchestrator for Phase 1: Flexibility vs. DDG Prediction Error.

Steps:
  1. Load and filter SKEMPI 2.0          (skempi.py)
  2. Fetch PDB structures                (structures.py)
  3. ANM flexibility scoring             (flexibility.py)
  4. FoldX DDG prediction                (foldx.py)
  5. Calibrate and compute error         (analysis.py)
  6. Statistics and plots                (analysis.py)
"""

import logging

from config import RESULTS_DIR, FIGURES_DIR, SKEMPI_CSV
from skempi import load_skempi
from structures import fetch_structures
from flexibility import assign_flexibility_to_mutations
from foldx import run_foldx_for_group
from analysis import calibrate_predictions, run_statistics, plot_results

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load SKEMPI
    skempi = load_skempi(SKEMPI_CSV)

    # 2. Fetch PDB structures
    pdb_paths = fetch_structures(skempi["pdb_id"].unique().tolist())

    # 3. Assign ANM flexibility scores to each mutation
    skempi = assign_flexibility_to_mutations(skempi, pdb_paths)

    # Keep only interface residues for the main analysis
    skempi_if = skempi[skempi["is_interface"] == True].copy()
    log.info("Interface mutations retained: %d", len(skempi_if))

    # 4. FoldX DDG predictions
    log.info("Running FoldX predictions...")
    all_ddg = []
    for pdb_id, group in skempi_if.groupby("pdb_id"):
        all_ddg.extend(run_foldx_for_group(pdb_id, group, pdb_paths))
    skempi_if["ddg_foldx"] = all_ddg

    # 5. Calibrate and compute prediction error
    skempi_if = calibrate_predictions(skempi_if)

    # 6. Statistics and plots
    results = run_statistics(skempi_if)
    plot_results(skempi_if)

    # Save enriched dataset
    out_csv = RESULTS_DIR / "mutations_with_flexibility_and_error.csv"
    skempi_if.to_csv(out_csv, index=False)
    log.info("Results saved to %s", out_csv)

    print("\n=== Phase 1 Results ===")
    print(f"  Mutations analysed : {results['n']}")
    print(f"  Spearman rho       : {results['spearman_rho']:.3f}")
    print(f"  Spearman p-value   : {results['spearman_p']:.4f}")
    print(f"  Mann-Whitney p     : {results['mw_p']:.4f}  (low vs high flex tertile)")


if __name__ == "__main__":
    main()
