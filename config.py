"""
config.py — Central configuration for the flexibility vs. DDG pipeline.
Edit these values before running.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FOLDX_BIN      = Path(os.environ.get("FOLDX_BIN", "/usr/local/bin/foldx"))
SKEMPI_CSV     = Path("data/skempi_v2.csv")
STRUCTURES_DIR = Path("data/structures")
FOLDX_WORK_DIR = Path("data/foldx_runs")
RESULTS_DIR    = Path("results")
FIGURES_DIR    = Path("results/figures")

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

RESOLUTION_CUTOFF = 2.5   # Angstrom — max crystal resolution to include
INTERFACE_CUTOFF  = 5.0   # Angstrom — residue counts as interface if within this distance of partner chain

# ---------------------------------------------------------------------------
# ANM
# ---------------------------------------------------------------------------

ANM_MODES = 10   # number of slowest modes to include in MSF calculation
