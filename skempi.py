"""
skempi.py — Load and filter SKEMPI 2.0.

Note: SKEMPI 2.0 does not include a Resolution column.
Resolution filtering is applied separately via fetch_resolution()
after PDB structures are downloaded.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def load_skempi(path: Path) -> pd.DataFrame:
    """
    Load SKEMPI 2.0 CSV and apply filters that don't require external data:
      - Single-point mutations only (no comma in mutation string)
      - Must have computable DDG from affinity values

    Adds parsed columns: pdb_id, chain, resnum, wt_aa, mut_aa, DDG
    """
    log.info("Loading SKEMPI from %s", path)
    df = pd.read_csv(path, sep=";")
    n_raw = len(df)

    # Single-point mutations only
    df = df[~df["Mutation(s)_PDB"].str.contains(",", na=True)].copy()

    # Compute DDG from affinity values (kcal/mol), drop rows where it can't be computed
    import numpy as np
    RT = 0.592   # kcal/mol at 298 K
    df["DDG"] = RT * np.log(
        df["Affinity_mut_parsed"].astype(float) /
        df["Affinity_wt_parsed"].astype(float)
    )
    df = df[df["DDG"].notna() & np.isfinite(df["DDG"])].copy()

    # Parse mutation fields from Mutation(s)_PDB which uses actual PDB
    # residue numbers and chain IDs — required for FoldX compatibility.
    # Mutation(s)_cleaned uses renumbered positions that don't match the PDB.
    df["pdb_id"] = df["#Pdb"].str[:4].str.upper()
    df["chain"]  = df["Mutation(s)_PDB"].str[1]
    df["wt_aa"]  = df["Mutation(s)_PDB"].str[0]
    df["mut_aa"] = df["Mutation(s)_PDB"].str[-1]
    # Strip insertion codes (e.g. '100b' -> 100) — keep only leading digits
    import re
    df["resnum_str"] = df["Mutation(s)_PDB"].str[2:-1]
    df["resnum"] = pd.to_numeric(df["resnum_str"].str.extract(r"^(\d+)")[0], errors="coerce")
    df = df[df["resnum"].notna()].copy()
    df["resnum"] = df["resnum"].astype(int)
    df["has_insertion_code"] = df["resnum_str"].str.contains(r"[A-Za-z]", regex=True)
    # Drop residues with insertion codes — FoldX handles them inconsistently
    n_before = len(df)
    df = df[~df["has_insertion_code"]].copy()
    log.info("Dropped %d mutations with insertion codes", n_before - len(df))

    log.info("SKEMPI: %d raw → %d after filtering", n_raw, len(df))
    return df.reset_index(drop=True)


def filter_by_resolution(
    df: pd.DataFrame,
    pdb_resolutions: dict[str, float],
    cutoff: float,
) -> pd.DataFrame:
    """
    Filter mutations to structures with resolution <= cutoff.

    pdb_resolutions: {pdb_id: resolution_angstrom} — build this with
    fetch_resolutions() from structures.py after downloading PDBs.
    """
    df = df.copy()
    df["resolution"] = df["pdb_id"].map(pdb_resolutions)
    before = len(df)
    df = df[df["resolution"] <= cutoff]
    log.info("Resolution filter (<=%.1fÅ): %d → %d mutations", cutoff, before, len(df))
    return df.reset_index(drop=True)


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/skempi_v2.csv")
    df = load_skempi(path)
    print(df[["pdb_id", "chain", "resnum", "wt_aa", "mut_aa", "DDG"]].head(10))
    print(f"\nTotal mutations: {len(df)}")
    print(f"Unique complexes: {df['pdb_id'].nunique()}")