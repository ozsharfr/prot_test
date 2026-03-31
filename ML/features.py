"""
features.py — Extract biophysical and structural features for each mutation.

Features computed:
  Flexibility (already in CSV):
    msf_z, msf_z_neighbors_2, msf_z_neighbors_4

  Mutation properties (from sequence):
    volume_change, hydrophobicity_change, charge_change
    is_to_gly, is_to_pro, is_to_ala, is_to_cys
    blosum62_score

  Structural context (from PDB):
    b_factor_wt          — crystallographic B-factor at mutated residue
    b_factor             — crystallographic B-factor at mutated residue
    dist_to_interface    — distance to partner chain centroid

  Interface location (from SKEMPI):
    location_INT, location_COR, location_RIM, location_SUR, location_SUP
    (one-hot encoding of iMutation_Location(s))
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Amino acid property lookup tables
# ---------------------------------------------------------------------------

# van der Waals volume (Å³) — Pontius et al. 1996
AA_VOLUME = {
    'A': 88.6,  'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'E': 138.4, 'Q': 143.8, 'G': 60.1,  'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0,  'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
}

# Kyte-Doolittle hydrophobicity scale
AA_HYDROPHOBICITY = {
    'A':  1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C':  2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I':  4.5,
    'L':  3.8, 'K': -3.9, 'M':  1.9, 'F':  2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V':  4.2,
}

# Formal charge at pH 7
AA_CHARGE = {
    'A':  0, 'R': +1, 'N':  0, 'D': -1, 'C':  0,
    'E': -1, 'Q':  0, 'G':  0, 'H':  0, 'I':  0,
    'L':  0, 'K': +1, 'M':  0, 'F':  0, 'P':  0,
    'S':  0, 'T':  0, 'W':  0, 'Y':  0, 'V':  0,
}

# BLOSUM62 matrix (symmetric, upper triangle stored as dict)
# Source: NCBI BLOSUM62
_BLOSUM62_RAW = {
    ('A','A'):4,('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0,('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1,('A','T'):0,('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5,('R','N'):0,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):0,('R','G'):-2,('R','H'):0,('R','I'):-3,('R','L'):-2,
    ('R','K'):2,('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6,('N','D'):1,('N','C'):-3,('N','Q'):0,('N','E'):0,
    ('N','G'):0,('N','H'):1,('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1,('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6,('D','C'):-3,('D','Q'):0,('D','E'):2,('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0,('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9,('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,
    ('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,
    ('C','V'):-1,
    ('Q','Q'):5,('Q','E'):2,('Q','G'):-2,('Q','H'):0,('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1,('Q','M'):0,('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0,('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5,('E','G'):-2,('E','H'):0,('E','I'):-3,('E','L'):-3,
    ('E','K'):1,('E','M'):-2,('E','F'):-3,('E','P'):-1,('E','S'):0,
    ('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6,('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-3,('G','P'):-2,('G','S'):0,('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8,('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2,('H','V'):-3,
    ('I','I'):4,('I','L'):2,('I','K'):-1,('I','M'):1,('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4,('L','K'):-2,('L','M'):2,('L','F'):0,('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5,('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5,('M','F'):0,('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6,('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3,('F','V'):-1,
    ('P','P'):7,('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,
    ('P','V'):-2,
    ('S','S'):4,('S','T'):1,('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5,('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2,('W','V'):-3,
    ('Y','Y'):7,('Y','V'):-1,
    ('V','V'):4,
}

def _blosum62(aa1: str, aa2: str) -> int:
    key = (aa1, aa2) if (aa1, aa2) in _BLOSUM62_RAW else (aa2, aa1)
    return _BLOSUM62_RAW.get(key, 0)


# ---------------------------------------------------------------------------
# Sequence-derived features
# ---------------------------------------------------------------------------

def add_mutation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add mutation property features derived purely from wt_aa and mut_aa.

    For continuous physicochemical changes we keep both the signed value
    (direction of change) and the absolute value (magnitude of disruption),
    since both carry biophysical meaning:
      - signed: e.g. charge gain vs charge loss have different structural effects
      - absolute: how large the perturbation is regardless of direction
    """
    df = df.copy()

    # Signed changes — direction matters
    df["volume_change"]         = df["mut_aa"].map(AA_VOLUME)         - df["wt_aa"].map(AA_VOLUME)
    df["hydrophobicity_change"] = df["mut_aa"].map(AA_HYDROPHOBICITY) - df["wt_aa"].map(AA_HYDROPHOBICITY)
    df["charge_change"]         = df["mut_aa"].map(AA_CHARGE)         - df["wt_aa"].map(AA_CHARGE)

    # Absolute changes — magnitude of disruption regardless of direction
    df["abs_volume_change"]         = df["volume_change"].abs()
    df["abs_hydrophobicity_change"] = df["hydrophobicity_change"].abs()
    df["abs_charge_change"]         = df["charge_change"].abs()

    # Flexibility — magnitude only (z-score sign is arbitrary, +2 and -2 are equally "unusual")
    for col in ["msf_z", "msf_z_neighbors_2", "msf_z_neighbors_4"]:
        if col in df.columns:
            df[f"abs_{col}"] = df[col].abs()

    # Binary mutation flags
    df["is_to_gly"]   = (df["mut_aa"] == "G").astype(int)
    df["is_to_pro"]   = (df["mut_aa"] == "P").astype(int)
    df["is_to_ala"]   = (df["mut_aa"] == "A").astype(int)
    df["is_to_cys"]   = (df["mut_aa"] == "C").astype(int)
    df["is_from_gly"] = (df["wt_aa"]  == "G").astype(int)
    df["is_from_pro"] = (df["wt_aa"]  == "P").astype(int)

    # BLOSUM62: negative = disruptive, so abs = substitution severity
    df["blosum62"]     = df.apply(lambda r: _blosum62(r["wt_aa"], r["mut_aa"]), axis=1)
    df["abs_blosum62"] = df["blosum62"].abs()

    return df


# ---------------------------------------------------------------------------
# Interface location one-hot
# ---------------------------------------------------------------------------

def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode iMutation_Location(s) column."""
    df = df.copy()
    loc_col = "iMutation_Location(s)"
    if loc_col not in df.columns:
        return df
    for loc in ["INT", "COR", "RIM", "SUR", "SUP"]:
        df[f"location_{loc}"] = df[loc_col].str.contains(loc, na=False).astype(int)
    return df


# ---------------------------------------------------------------------------
# Structural features from PDB
# ---------------------------------------------------------------------------

def add_structural_features(df: pd.DataFrame, pdb_paths: dict) -> pd.DataFrame:
    """
    Add B-factor and distance to interface centroid from PDB structure files.
    Requires: biopython, numpy
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    df = df.copy()

    b_factors  = []
    dist_to_if = []

    for _, row in df.iterrows():
        pdb_path = pdb_paths.get(row["pdb_id"])
        if pdb_path is None or not Path(pdb_path).exists():
            b_factors.append(np.nan)
            dist_to_if.append(np.nan)
            continue

        try:
            structure = parser.get_structure(row["pdb_id"], str(pdb_path))
            model     = structure[0]
            chain     = model[row["chain"]]
            resnum    = int(row["resnum"])

            # B-factor — mean over backbone atoms of the mutated residue
            try:
                residue = chain[resnum]
                bf = np.mean([a.get_bfactor() for a in residue.get_atoms()
                              if a.get_name() in ("CA", "N", "C", "O")])
                b_factors.append(float(bf))
            except (KeyError, ValueError):
                b_factors.append(np.nan)

            # Distance to interface centroid (partner chain Cα centroid)
            try:
                partner_chains = [c for c in model.get_chains()
                                  if c.id != row["chain"]]
                if partner_chains:
                    partner_coords = np.array([
                        a.get_vector().get_array()
                        for pc in partner_chains
                        for r in pc.get_residues()
                        for a in r.get_atoms()
                        if a.get_name() == "CA"
                    ])
                    centroid = partner_coords.mean(axis=0)
                    res_ca   = chain[resnum]["CA"].get_vector().get_array()
                    dist_to_if.append(float(np.linalg.norm(res_ca - centroid)))
                else:
                    dist_to_if.append(np.nan)
            except Exception:
                dist_to_if.append(np.nan)

        except Exception as e:
            log.debug("Structural features failed for %s/%s%d: %s",
                      row["pdb_id"], row["chain"], row["resnum"], e)
            b_factors.append(np.nan)
            dist_to_if.append(np.nan)

    df["b_factor"]          = b_factors
    df["dist_to_interface"] = dist_to_if
    return df


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Flexibility
    "msf_z", "msf_z_neighbors_2", "msf_z_neighbors_4",
    # Mutation properties
    "volume_change", "hydrophobicity_change", "charge_change",
    "is_to_gly", "is_to_pro", "is_to_ala", "is_to_cys",
    "is_from_gly", "is_from_pro", "blosum62",
    # Interface location
    "location_INT", "location_COR", "location_RIM", "location_SUR", "location_SUP",
    # Structural
    "b_factor", "dist_to_interface",
]


def build_features(df: pd.DataFrame, pdb_paths: dict = None) -> pd.DataFrame:
    """
    Run all feature extractors and return enriched DataFrame.
    pdb_paths is optional — if None, structural and protein-level features are skipped.
    """
    df = add_mutation_features(df)
    df = add_location_features(df)
    if pdb_paths is not None:
        df = add_structural_features(df, pdb_paths)
        df = add_protein_features(df, pdb_paths)
    return df


# ---------------------------------------------------------------------------
# Protein-level features (one row per pdb_id, then joined to mutations)
# ---------------------------------------------------------------------------

def compute_protein_features(pdb_id: str, pdb_path: Path) -> dict:
    """
    Compute protein-level structural features from a PDB file.

    Returns a dict with keys prefixed by 'prot_' so they're easy to
    identify in the feature matrix.
    """
    feats = {"pdb_id": pdb_id}

    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, str(pdb_path))
        model = structure[0]
        chains = list(model.get_chains())

        # --- Basic counts ---
        all_residues = [r for c in chains for r in c.get_residues()
                        if r.get_id()[0] == ' ']  # exclude HETATM
        feats["prot_n_chains"]   = len(chains)
        feats["prot_n_residues"] = len(all_residues)

        # --- B-factor stats (whole protein and per chain) ---
        all_bfactors = []
        for r in all_residues:
            for a in r.get_atoms():
                if a.get_name() == 'CA':
                    all_bfactors.append(a.get_bfactor())

        if all_bfactors:
            feats["prot_mean_bfactor"] = float(np.mean(all_bfactors))
            feats["prot_std_bfactor"]  = float(np.std(all_bfactors))
            feats["prot_max_bfactor"]  = float(np.max(all_bfactors))

        # --- Interface size ---
        # Count residues within 8Å of any other chain (Cα-Cα)
        try:
            interface_count = 0
            for chain in chains:
                chain_ca = np.array([
                    r["CA"].get_vector().get_array()
                    for r in chain.get_residues()
                    if r.get_id()[0] == ' ' and "CA" in r
                ])
                partner_ca = np.array([
                    r["CA"].get_vector().get_array()
                    for c in chains if c.id != chain.id
                    for r in c.get_residues()
                    if r.get_id()[0] == ' ' and "CA" in r
                ])
                if len(chain_ca) == 0 or len(partner_ca) == 0:
                    continue
                for coord in chain_ca:
                    dists = np.linalg.norm(partner_ca - coord, axis=1)
                    if np.any(dists <= 8.0):
                        interface_count += 1
            feats["prot_n_interface_residues"] = interface_count
            if feats["prot_n_residues"] > 0:
                feats["prot_frac_interface"] = interface_count / feats["prot_n_residues"]
        except Exception as e:
            log.debug("Interface count failed for %s: %s", pdb_id, e)

    except Exception as e:
        log.warning("Protein features failed for %s: %s", pdb_id, e)

    return feats


def add_protein_features(df: pd.DataFrame, pdb_paths: dict) -> pd.DataFrame:
    """
    Compute protein-level features for each unique pdb_id and join to df.
    Each mutation gets the features of its parent complex.
    """
    if not pdb_paths:
        log.warning("No PDB paths provided — skipping protein-level features")
        return df

    prot_rows = []
    for pdb_id, pdb_path in pdb_paths.items():
        if Path(pdb_path).exists():
            log.info("Computing protein features for %s", pdb_id)
            prot_rows.append(compute_protein_features(pdb_id, Path(pdb_path)))

    if not prot_rows:
        return df

    prot_df = pd.DataFrame(prot_rows)
    df = df.merge(prot_df, on="pdb_id", how="left")
    log.info("Added %d protein-level features", len(prot_df.columns) - 1)
    return df