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
FOLDX_BIN = Path(r"C:\Users\ozsha\Downloads\foldxWindows\foldx_20261231.exe")
SKEMPI_CSV     = Path("data/skempi_v2.csv")
STRUCTURES_DIR = Path("data/structures")
FOLDX_WORK_DIR = Path("data/foldx_runs")
RESULTS_DIR    = Path("results")
FIGURES_DIR    = Path("results/figures")

PILOT_PDB_IDS =['3SGB' , '3S9D','1JTG' ,'1AO7','2FTL','1A22','1JRH', '1CHO', '1R0R', '1PPF', 
                '3BT1', '3SE3','2NZ9','3HFM','1BRS','4BFI','2WPT','2JEL','1CBW',
                '4RS1','3EQS','3MZG','1DAN','3QDG','3BN9']
    # "1CSE",   # Subtilisin / eglin c — many SKEMPI entries, classic benchmark
    # "1VFB",   # Antibody / lysozyme — good interface, canonical test case
    # "1A22",   # Human growth hormone / receptor — well-studied, good mutation coverage




# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

RESOLUTION_CUTOFF = 4.0   # Angstrom — max crystal resolution to include
INTERFACE_CUTOFF  = 12.0   # Angstrom — residue counts as interface if within this distance of partner chain

# ---------------------------------------------------------------------------
# ANM
# ---------------------------------------------------------------------------

ANM_MODES = 10   # number of slowest modes to include in MSF calculation
