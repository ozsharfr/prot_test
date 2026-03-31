"""
structures.py — Fetch PDB structures from RCSB.
"""

import logging
import requests
from pathlib import Path

from config import STRUCTURES_DIR

log = logging.getLogger(__name__)

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def _download_pdb(pdb_id: str, dest: Path):
    """Download a PDB file directly from RCSB in PDB format."""
    url = RCSB_PDB_URL.format(pdb_id=pdb_id)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.write_text(r.text)


def fetch_structures(pdb_ids: list[str]) -> dict[str, Path]:
    """
    Download PDB files for each unique complex ID directly from RCSB.
    Skips structures already present on disk.

    Returns: {pdb_id: local Path}
    """
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}

    for pdb_id in set(pdb_ids):
        dest = STRUCTURES_DIR / f"{pdb_id.lower()}.pdb"

        if dest.exists() and dest.stat().st_size > 0:
            paths[pdb_id] = dest
            continue

        log.info("Fetching %s from RCSB...", pdb_id)
        try:
            _download_pdb(pdb_id, dest)
            paths[pdb_id] = dest
            log.info("Saved %s → %s", pdb_id, dest)
        except requests.HTTPError as e:
            log.warning("Could not fetch %s: %s", pdb_id, e)
            if dest.exists():
                dest.unlink()   # remove empty/partial file
        except Exception as e:
            log.warning("Unexpected error fetching %s: %s", pdb_id, e)
            if dest.exists():
                dest.unlink()

    return paths


if __name__ == "__main__":
    # Quick test: fetch a few structures
    # 3C4D does not exist in PDB — should log a warning and be absent from result
    test_ids = ["1A22", "2B3E", "3C4D"]
    fetched = fetch_structures(test_ids)
    print(f"Fetched structures: {fetched}")


def fetch_resolutions(pdb_ids: list[str]) -> dict[str, float]:
    """
    Fetch crystal resolution (Å) for each PDB ID from RCSB's REST API.
    Returns {pdb_id: resolution}. Missing entries are omitted.
    """
    import json
    resolutions = {}
    for pdb_id in pdb_ids:
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            res = (
                data.get("refine", [{}])[0].get("ls_d_res_high") or
                data.get("reflns", [{}])[0].get("d_resolution_high")
            )
            if res is not None:
                resolutions[pdb_id] = float(res)
        except Exception as e:
            log.debug("Could not fetch resolution for %s: %s", pdb_id, e)
    return resolutions
