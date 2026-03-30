"""
flexibility.py — ANM-based per-residue flexibility scoring via ProDy.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import prody
from prody.measure import findNeighbors

from config import ANM_MODES, INTERFACE_CUTOFF

log = logging.getLogger(__name__)


def get_interface_residues(structure: prody.AtomGroup, chain: str) -> set[int]:
    """
    Return residue numbers in `chain` that are within INTERFACE_CUTOFF Å
    of any atom in any other chain.
    """
    chain_sel   = structure.select(f"chain {chain} and calpha")
    partner_sel = structure.select(f"not chain {chain} and calpha")
    if chain_sel is None or partner_sel is None:
        return set()

    contacts = findNeighbors(chain_sel, INTERFACE_CUTOFF, partner_sel)
    return {pair[0].getResnum() for pair in contacts}


def compute_anm_msf(pdb_path: Path, chain: str) -> pd.DataFrame:
    """
    Run ANM on the full complex (Cα only) and return per-residue flexibility
    scores for residues in the specified chain.

    ANM is run on the whole complex so that interface rigidification caused
    by the partner chain is captured in the flexibility estimate.

    Returns DataFrame with columns:
        resnum        — residue number
        msf           — raw mean square fluctuation
        msf_z         — z-score normalized MSF within the chain
        is_interface  — whether residue is within INTERFACE_CUTOFF of partner
    """
    log.info("Running ANM on %s (chain %s)", pdb_path.name, chain)

    structure = prody.parsePDB(str(pdb_path), model=1)
    if structure is None:
        raise ValueError(f"Could not parse {pdb_path}")

    calpha = structure.select("calpha and protein")
    if calpha is None:
        raise ValueError(f"No Cα atoms found in {pdb_path}")

    anm = prody.ANM(str(pdb_path))
    anm.buildHessian(calpha)
    anm.calcModes(n_modes=ANM_MODES)

    msf_values = prody.calcSqFlucts(anm)
    interface_resnums = get_interface_residues(structure, chain)

    records = []
    for i, (resnum, msf) in enumerate(zip(calpha.getResnums(), msf_values)):
        if calpha.getChids()[i] != chain:
            continue
        records.append({
            "resnum":       resnum,
            "msf":          msf,
            "is_interface": resnum in interface_resnums,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["msf_z"] = stats.zscore(df["msf"]) if len(df) > 1 else 0.0
    return df


def assign_flexibility_to_mutations(
    skempi: pd.DataFrame,
    pdb_paths: dict[str, Path],
) -> pd.DataFrame:
    """
    For each mutation in skempi, look up the ANM MSF of the mutated residue.
    Caches ANM results per (pdb_id, chain) to avoid redundant computation.

    Adds columns: msf, msf_z, is_interface
    """
    cache: dict[str, pd.DataFrame] = {}

    def _lookup(row: pd.Series) -> pd.Series:
        key = f"{row['pdb_id']}_{row['chain']}"
        if key not in cache:
            pdb_path = pdb_paths.get(row["pdb_id"])
            if pdb_path is None or not pdb_path.exists():
                cache[key] = pd.DataFrame(columns=["resnum", "msf", "msf_z", "is_interface"])
            else:
                try:
                    cache[key] = compute_anm_msf(pdb_path, row["chain"])
                except Exception as e:
                    log.warning("ANM failed for %s: %s", key, e)
                    cache[key] = pd.DataFrame(columns=["resnum", "msf", "msf_z", "is_interface"])

        match = cache[key][cache[key]["resnum"] == row["resnum"]]
        if match.empty:
            return pd.Series({"msf": float("nan"), "msf_z": float("nan"), "is_interface": False})
        return match.iloc[0][["msf", "msf_z", "is_interface"]]

    log.info("Assigning ANM flexibility scores to %d mutations...", len(skempi))
    msf_cols = skempi.apply(_lookup, axis=1)
    return pd.concat([skempi, msf_cols], axis=1)
