"""
skempi.py — Load and filter SKEMPI 2.0.
"""

import logging
from pathlib import Path

import pandas as pd

from config import RESOLUTION_CUTOFF

log = logging.getLogger(__name__)


def load_skempi(path: Path) -> pd.DataFrame:
    """
    Load SKEMPI 2.0 CSV and apply standard filters:
      - Single-point mutations only
      - Resolution <= RESOLUTION_CUTOFF
      - Must have experimental DDG value

    Adds parsed columns: pdb_id, chain, resnum, wt_aa, mut_aa
    """
    log.info("Loading SKEMPI from %s", path)
    df = pd.read_csv(path, sep=";")
    n_raw = len(df)

    df = df[~df["Mutation(s)_cleaned"].str.contains(",", na=True)]
    df = df[df["Resolution"] <= RESOLUTION_CUTOFF]
    df = df[df["DDG"].notna()].copy()

    df["pdb_id"] = df["#Pdb"].str[:4].str.upper()
    df["chain"]  = df["Mutation(s)_cleaned"].str[1]
    df["resnum"] = df["Mutation(s)_cleaned"].str[2:-1].astype(int)
    df["wt_aa"]  = df["Mutation(s)_cleaned"].str[0]
    df["mut_aa"] = df["Mutation(s)_cleaned"].str[-1]

    log.info("SKEMPI: %d raw → %d after filtering", n_raw, len(df))
    return df.reset_index(drop=True)
