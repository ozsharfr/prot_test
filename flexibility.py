"""
flexibility.py — ANM-based per-residue flexibility scoring via ProDy.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import prody
from config import ANM_MODES, INTERFACE_CUTOFF

log = logging.getLogger(__name__)

# Suppress ProDy's verbose output
prody.confProDy(verbosity="none")


def get_interface_residues(structure: prody.AtomGroup, chain: str) -> set[int]:
    """
    Return residue numbers in `chain` that are within INTERFACE_CUTOFF Å
    of any atom in any other chain.
    Uses pure NumPy distance calculation to avoid ProDy KDTree dtype bug on Windows.
    """
    chain_sel   = structure.select(f"chain {chain} and calpha")
    partner_sel = structure.select(f"not chain {chain} and calpha")
    if chain_sel is None or partner_sel is None:
        log.warning("Could not select chain %s or partner — no interface residues found", chain)
        return set()

    chain_coords   = chain_sel.getCoords().astype("float64")    # (N, 3)
    partner_coords = partner_sel.getCoords().astype("float64")  # (M, 3)
    chain_resnums  = chain_sel.getResnums()

    interface = set()
    for i, coord in enumerate(chain_coords):
        diffs = partner_coords - coord          # (M, 3)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        if np.any(dists <= INTERFACE_CUTOFF):
            interface.add(int(chain_resnums[i]))
    return interface


def compute_anm_msf(pdb_path: Path, chain: str) -> pd.DataFrame:
    """
    Run ANM on the full complex (Cα only) and return per-residue flexibility
    scores for residues in the specified chain.

    Returns DataFrame with columns: resnum, msf, msf_z, is_interface
    """
    log.info("Running ANM on %s (chain %s)", pdb_path.name, chain)

    structure = prody.parsePDB(str(pdb_path), model=1)
    if structure is None:
        raise ValueError(f"Could not parse {pdb_path}")

    calpha = structure.select("calpha and protein")
    if calpha is None:
        raise ValueError(f"No Cα atoms found in {pdb_path}")

    log.info("  Cα atoms: %d across chains: %s",
             calpha.numAtoms(), list(set(calpha.getChids())))

    anm = prody.ANM(str(pdb_path))
    anm.buildHessian(calpha)
    anm.calcModes(n_modes=ANM_MODES)
    msf_values = prody.calcSqFlucts(anm)
    interface_resnums = get_interface_residues(structure, chain)
    log.info("  Interface residues in chain %s: %d", chain, len(interface_resnums))

    records = []
    for i, (resnum, msf) in enumerate(zip(calpha.getResnums(), msf_values)):
        if calpha.getChids()[i] != chain:
            continue
        records.append({
            "resnum":       int(resnum),
            "msf":          float(msf),
            "is_interface": resnum in interface_resnums,
        })

    if not records:
        log.warning("  No residues found for chain %s in %s", chain, pdb_path.name)
        return pd.DataFrame(columns=["resnum", "msf", "msf_z", "is_interface"])

    df = pd.DataFrame(records)
    df["msf_z"] = stats.zscore(df["msf"]) if len(df) > 1 else 0.0
    log.info("  Processed %d residues (%d interface)",
             len(df), df["is_interface"].sum())
    return df


def _neighbor_msf(anm_df: pd.DataFrame, resnum: int, window: int) -> float:
    """
    Average msf_z of residues within +/- window positions of resnum
    (excluding the residue itself). Uses sequential position index,
    not residue number, to handle gaps in numbering correctly.
    """
    if anm_df.empty:
        return np.nan
    # Find the positional index of this residue
    idx_matches = anm_df.index[anm_df["resnum"] == resnum].tolist()
    if not idx_matches:
        return np.nan
    pos = anm_df.index.get_loc(idx_matches[0])
    lo  = max(0, pos - window)
    hi  = min(len(anm_df), pos + window + 1)
    neighbors = anm_df.iloc[lo:hi]
    neighbors = neighbors[neighbors["resnum"] != resnum]
    return float(neighbors["msf_z"].mean()) if len(neighbors) > 0 else np.nan


def assign_flexibility_to_mutations(
    skempi: pd.DataFrame,
    pdb_paths: dict[str, Path],
    neighbor_windows: list[int] = [2, 4],
) -> pd.DataFrame:
    """
    For each mutation in skempi, look up the ANM MSF of the mutated residue
    and compute neighborhood flexibility scores.
    Caches ANM results per (pdb_id, chain) to avoid redundant computation.

    Adds columns:
        msf, msf_z          — residue-level flexibility
        is_interface        — within INTERFACE_CUTOFF of partner chain
        msf_z_neighbors_2   — mean msf_z of ±2 sequence neighbors
        msf_z_neighbors_4   — mean msf_z of ±4 sequence neighbors
    """
    cache: dict[str, pd.DataFrame] = {}

    msf_list          = []
    msf_z_list        = []
    is_interface_list = []
    neighbor_lists    = {w: [] for w in neighbor_windows}

    for _, row in skempi.iterrows():
        key = f"{row['pdb_id']}_{row['chain']}"

        if key not in cache:
            pdb_path = pdb_paths.get(row["pdb_id"])
            if pdb_path is None or not pdb_path.exists():
                log.warning("No PDB file for %s — skipping ANM", row["pdb_id"])
                cache[key] = pd.DataFrame(columns=["resnum", "msf", "msf_z", "is_interface"])
            else:
                try:
                    cache[key] = compute_anm_msf(pdb_path, row["chain"])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    log.warning("ANM failed for %s: %s", key, e)
                    cache[key] = pd.DataFrame(columns=["resnum", "msf", "msf_z", "is_interface"])

        anm_df = cache[key]
        match  = anm_df[anm_df["resnum"] == row["resnum"]]

        if match.empty:
            msf_list.append(np.nan)
            msf_z_list.append(np.nan)
            is_interface_list.append(False)
            for w in neighbor_windows:
                neighbor_lists[w].append(np.nan)
        else:
            msf_list.append(match.iloc[0]["msf"])
            msf_z_list.append(match.iloc[0]["msf_z"])
            is_interface_list.append(bool(match.iloc[0]["is_interface"]))
            for w in neighbor_windows:
                neighbor_lists[w].append(_neighbor_msf(anm_df, row["resnum"], w))

    skempi = skempi.copy()
    skempi["msf"]          = msf_list
    skempi["msf_z"]        = msf_z_list
    skempi["is_interface"] = is_interface_list
    for w in neighbor_windows:
        skempi[f"msf_z_neighbors_{w}"] = neighbor_lists[w]

    log.info("Flexibility assigned: %d/%d mutations have ANM scores, %d at interface",
             skempi["msf"].notna().sum(), len(skempi), skempi["is_interface"].sum())

    return skempi
