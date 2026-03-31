"""
find_candidates.py — Find best SKEMPI complexes for the flexibility vs. DDG study.

Criteria:
  1. Has single-point mutations in SKEMPI
  2. Enough mutations per complex (>= MIN_MUTATIONS)
  3. Good crystal resolution (<= RESOLUTION_CUTOFF)
  4. PDB structure is available
  5. Has flexible regions (high B-factor variance as a proxy before ANM)
"""

import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SKEMPI_CSV       = Path("../data/skempi_v2.csv")
MIN_MUTATIONS    = 10     # minimum single-point mutations per complex
RESOLUTION_MAX   = 6.0   # Angstrom
TOP_N            = 25    # how many candidates to show

# ---------------------------------------------------------------------------
# Load and filter SKEMPI
# ---------------------------------------------------------------------------
log.info("Loading SKEMPI...")
df = pd.read_csv(SKEMPI_CSV, sep=";")

# Single-point mutations only
single = df[~df["Mutation(s)_PDB"].str.contains(",", na=True)].copy()

# Must have computable DDG
single = single[single["Affinity_mut_parsed"].notna() & 
                single["Affinity_wt_parsed"].notna()].copy()

RT = 0.592
single["DDG"] = RT * np.log(
    single["Affinity_mut_parsed"].astype(float) /
    single["Affinity_wt_parsed"].astype(float)
)
single = single[single["DDG"].notna() & np.isfinite(single["DDG"])].copy()

# Parse PDB ID
single["pdb_id"] = single["#Pdb"].str[:4].str.upper()

log.info("Single-point mutations with DDG: %d across %d complexes",
         len(single), single["pdb_id"].nunique())

# ---------------------------------------------------------------------------
# Count mutations per complex
# ---------------------------------------------------------------------------
counts = single.groupby("pdb_id").agg(
    n_mutations   = ("DDG", "count"),
    ddg_std       = ("DDG", "std"),    # spread of DDG values — more = more informative
    ddg_mean      = ("DDG", "mean"),
).reset_index()

candidates = counts[counts["n_mutations"] >= MIN_MUTATIONS].copy()
log.info("Complexes with >= %d single mutations: %d", MIN_MUTATIONS, len(candidates))

# ---------------------------------------------------------------------------
# Fetch resolutions from RCSB
# ---------------------------------------------------------------------------
log.info("Fetching resolutions for %d complexes...", len(candidates))

resolutions = {}
for pdb_id in candidates["pdb_id"].tolist():
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        res = (data.get("refine") or [{}])[0].get("ls_dres_high")
        if res is None:
            res = (data.get("rcsb_entry_info") or {}).get(
                "diffrn_resolution_high", {}).get("value")
        if res is not None:
            resolutions[pdb_id] = float(res)
    except Exception as e:
        log.debug("Could not fetch resolution for %s: %s", pdb_id, e)

log.info("Got resolutions for %d/%d complexes", len(resolutions), len(candidates))

candidates["resolution"] = candidates["pdb_id"].map(resolutions)
candidates = candidates[candidates["resolution"].notna()].copy()
candidates = candidates[candidates["resolution"] <= RESOLUTION_MAX].copy()
log.info("After resolution filter (<= %.1fÅ): %d complexes", RESOLUTION_MAX, len(candidates))

# ---------------------------------------------------------------------------
# Fetch B-factor variance as flexibility proxy (from RCSB summary stats)
# ---------------------------------------------------------------------------
log.info("Fetching B-factor info...")

bfactor_info = {}
for pdb_id in candidates["pdb_id"].tolist():
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        # Mean B-factor from refinement — higher = more flexible overall
        biso = (data.get("refine") or [{}])[0].get("biso_mean")
        if biso is not None:
            bfactor_info[pdb_id] = float(biso)
    except Exception:
        pass

candidates["mean_bfactor"] = candidates["pdb_id"].map(bfactor_info)

# ---------------------------------------------------------------------------
# Score and rank candidates
# ---------------------------------------------------------------------------
# Normalize each criterion to 0-1 and combine
def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

candidates["score"] = (
    norm(candidates["n_mutations"]) * 0.4 +   # more mutations = better stats
    norm(candidates["ddg_std"]) * 0.3 +        # more DDG spread = more informative
    norm(candidates["mean_bfactor"].fillna(0)) * 0.3  # higher B = more flexible
)

candidates = candidates.sort_values("score", ascending=False)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'='*75}")
print(f"Top {TOP_N} SKEMPI complexes for flexibility vs. DDG study")
print(f"{'='*75}")
print(f"{'PDB':<8} {'N_mut':>6} {'Res(Å)':>7} {'DDG_std':>8} {'Mean_B':>8} {'Score':>7}")
print(f"{'-'*75}")
for _, row in candidates.head(TOP_N).iterrows():
    bfac = f"{row['mean_bfactor']:.1f}" if pd.notna(row["mean_bfactor"]) else "  N/A"
    print(f"{row['pdb_id']:<8} {row['n_mutations']:>6} {row['resolution']:>7.2f} "
          f"{row['ddg_std']:>8.3f} {bfac:>8} {row['score']:>7.3f}")

print(f"\nTo use these in your pipeline, update config.py:")
top_ids = candidates.head(5)["pdb_id"].tolist()
print(f"PILOT_PDB_IDS = {top_ids}")