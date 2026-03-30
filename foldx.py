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


def _parse_foldx_ddg(raw_file: Path) -> float | None:
    """Extract DDG from a FoldX Raw_BuildModel .fxout output file."""
    if not raw_file.exists():
        log.warning("FoldX output not found: %s", raw_file)
        return None

    with open(raw_file) as f:
        for line in f:
            if line.startswith("Total"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    continue
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mutation_string(row: pd.Series) -> str:
    """Convert a SKEMPI row to FoldX mutation string: e.g. RA45A"""
    return f"{row['wt_aa']}{row['chain']}{row['resnum']}{row['mut_aa']}"


def repair_pdb(pdb_path: Path) -> Path:
    """
    Run FoldX RepairPDB on a structure.
    Returns path to the repaired PDB (cached if already exists).
    """
    repaired = FOLDX_WORK_DIR / f"{pdb_path.stem}_Repair.pdb"
    if repaired.exists():
        return repaired

    log.info("FoldX RepairPDB: %s", pdb_path.name)
    FOLDX_WORK_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [str(FOLDX_BIN), "--command=RepairPDB",
         f"--pdb={pdb_path.name}",
         f"--output-dir={FOLDX_WORK_DIR}"],
        cwd=str(pdb_path.parent),
        check=True,
        capture_output=True,
    )
    return repaired


def predict_ddg(repaired_pdb: Path, mut_str: str) -> float | None:
    """
    Run FoldX BuildModel for a single mutation.

    Args:
        repaired_pdb: path to the RepairPDB output
        mut_str:      FoldX mutation string, e.g. "RA45A"

    Returns:
        Predicted DDG in kcal/mol, or None if FoldX fails.
    """
    mut_dir = FOLDX_WORK_DIR / mut_str
    mut_dir.mkdir(parents=True, exist_ok=True)

    mutant_file = mut_dir / "individual_list.txt"
    _write_mutant_file(mut_str, mutant_file)

    try:
        subprocess.run(
            [str(FOLDX_BIN), "--command=BuildModel",
             f"--pdb={repaired_pdb.name}",
             f"--mutant-file={mutant_file}",
             f"--output-dir={mut_dir}"],
            cwd=str(repaired_pdb.parent),
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("FoldX BuildModel failed for %s: %s", mut_str, e)
        return None

    raw_file = mut_dir / f"Raw_{repaired_pdb.stem}.fxout"
    return _parse_foldx_ddg(raw_file)


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
