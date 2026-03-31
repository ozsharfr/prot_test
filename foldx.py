"""
foldx.py — FoldX RepairPDB + BuildModel wrappers for DDG prediction.
"""

import logging
import subprocess
from pathlib import Path

import pandas as pd

from config import FOLDX_BIN, FOLDX_WORK_DIR

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_mutant_file(mutation_str: str, path: Path):
    """Write a FoldX individual_list.txt for a single mutation."""
    with open(path, "w") as f:
        f.write(mutation_str + ";\n")


def _parse_foldx_ddg(fxout_file: Path) -> float | None:
    """
    Extract DDG from a FoldX Average_.fxout file.

    Format (tab-separated):
      col 0: PDB name (e.g. 1a22_Repair_1)
      col 1: SD
      col 2: total energy — this is already DDG (mutant - WT)

    Each data row is one mutation replicate. We average across replicates.
    Header/footer lines are skipped (non-numeric col 2).
    """
    if not fxout_file.exists():
        log.warning("FoldX output not found: %s", fxout_file)
        return None

    ddg_values = []
    with open(fxout_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                ddg_values.append(float(parts[2]))
            except ValueError:
                continue  # skip header lines

    if not ddg_values:
        log.warning("Could not parse DDG values from %s", fxout_file)
        return None

    ddg = sum(ddg_values) / len(ddg_values)
    log.debug("DDG=%.3f (n_replicates=%d)", ddg, len(ddg_values))
    return ddg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mutation_string(row: pd.Series) -> str:
    """Convert a SKEMPI row to FoldX mutation string: e.g. RA45A"""
    return f"{row['wt_aa']}{row['chain']}{row['resnum']}{row['mut_aa']}"


def repair_pdb(pdb_path: Path) -> Path:
    """
    Run FoldX RepairPDB on a structure.
    FoldX must run from the directory containing the PDB file,
    so we copy the PDB into FOLDX_WORK_DIR and run from there.
    Returns path to the repaired PDB (cached if already exists).
    """
    FOLDX_WORK_DIR.mkdir(parents=True, exist_ok=True)
    repaired = FOLDX_WORK_DIR / f"{pdb_path.stem}_Repair.pdb"
    if repaired.exists():
        return repaired

    import shutil

    # FoldX requires rotabase.txt in its working directory
    # Copy it from the FoldX installation folder if not already present
    rotabase_src = FOLDX_BIN.parent / "rotabase.txt"
    rotabase_dst = FOLDX_WORK_DIR / "rotabase.txt"
    if rotabase_src.exists() and not rotabase_dst.exists():
        shutil.copy2(rotabase_src, rotabase_dst)
        log.info("Copied rotabase.txt to work dir")
    elif not rotabase_src.exists() and not rotabase_dst.exists():
        log.warning("rotabase.txt not found in %s — FoldX may fail", FOLDX_BIN.parent)

    # Copy PDB into work dir so FoldX can find it
    local_pdb = FOLDX_WORK_DIR / pdb_path.name
    shutil.copy2(pdb_path, local_pdb)

    log.info("FoldX RepairPDB: %s", pdb_path.name)
    result = subprocess.run(
        [str(FOLDX_BIN), "--command=RepairPDB",
         f"--pdb={pdb_path.name}"],
        cwd=str(FOLDX_WORK_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("FoldX stdout: %s", result.stdout[-500:])
        log.error("FoldX stderr: %s", result.stderr[-500:])
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return repaired


def predict_ddg(repaired_pdb: Path, mut_str: str) -> float | None:
    """
    Run FoldX BuildModel for a single mutation.
    Runs from FOLDX_WORK_DIR so FoldX can find rotabase.txt and the repaired PDB.
    Output files are written to FOLDX_WORK_DIR directly.

    Args:
        repaired_pdb: path to the RepairPDB output (must be in FOLDX_WORK_DIR)
        mut_str:      FoldX mutation string, e.g. "RA45A"

    Returns:
        Predicted DDG in kcal/mol, or None if FoldX fails.
    """
    mut_dir = FOLDX_WORK_DIR / mut_str
    mut_dir.mkdir(parents=True, exist_ok=True)

    # Write mutant file in FOLDX_WORK_DIR (same dir FoldX runs from)
    # so the relative path always works on Windows
    mutant_file = FOLDX_WORK_DIR / "individual_list.txt"
    _write_mutant_file(mut_str, mutant_file)

    try:
        result = subprocess.run(
            [str(FOLDX_BIN), "--command=BuildModel",
             f"--pdb={repaired_pdb.name}",
             "--mutant-file=individual_list.txt"],  # relative path — safe on all OS
            cwd=str(FOLDX_WORK_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        log.debug("FoldX BuildModel stdout: %s", result.stdout[-500:])
        if result.returncode != 0:
            log.warning("FoldX BuildModel failed for %s (exit %d) stdout: %s",
                        mut_str, result.returncode, result.stdout[-500:])
            return None
    except subprocess.TimeoutExpired:
        log.warning("FoldX BuildModel timed out for %s", mut_str)
        return None

    # FoldX writes Average_ (always) and Raw_ (only with numberOfRuns>1)
    # Use Average_ as primary since it is always present
    import shutil
    stem = repaired_pdb.stem
    avg_file = FOLDX_WORK_DIR / f"Average_{stem}.fxout"
    raw_file = FOLDX_WORK_DIR / f"Raw_{stem}.fxout"

    fxout = avg_file if avg_file.exists() else raw_file if raw_file.exists() else None
    if fxout is None:
        all_fxout = [f.name for f in FOLDX_WORK_DIR.glob("*.fxout")]
        log.warning("No usable .fxout for %s. Present: %s", mut_str, all_fxout)
        return None

    ddg = _parse_foldx_ddg(fxout)
    # Archive parsed file to mut_dir so next mutation gets a fresh one
    shutil.move(str(fxout), str(mut_dir / fxout.name))
    # Also archive other fxout files produced for this run
    for f in FOLDX_WORK_DIR.glob(f"*_{stem}.fxout"):
        shutil.move(str(f), str(mut_dir / f.name))
    return ddg


def run_foldx_for_group(
    pdb_id: str,
    group: pd.DataFrame,
    pdb_paths: dict,
) -> list[float | None]:
    """
    Run RepairPDB once per complex, then BuildModel for each mutation in group.
    Returns a list of DDG values (or None) aligned to group rows.
    """
    pdb_path = pdb_paths.get(pdb_id)
    if pdb_path is None or not pdb_path.exists():
        return [None] * len(group)

    try:
        repaired = repair_pdb(pdb_path)
    except subprocess.CalledProcessError as e:
        log.warning("RepairPDB failed for %s: %s", pdb_id, e)
        return [None] * len(group)

    results = []
    for _, row in group.iterrows():
        mut_str = mutation_string(row)
        results.append(predict_ddg(repaired, mut_str))
    return results