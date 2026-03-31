# Beyond the Static Structure
## Predicting Functional Shifts in Protein Ensembles

A computational pipeline for testing the hypothesis that **conformational flexibility is the missing link in ΔΔG prediction**: regions of proteins that are structurally flexible tend to be predicted less accurately by static energy models like FoldX.

---

## Scientific Goal

Standard binding affinity predictors (FoldX, Rosetta) assume a single rigid structure. This project tests whether prediction error correlates with structural flexibility, using ANM-derived flexibility scores and crystallographic B-factors as proxies.

**Core hypothesis:**  
*Mutations at flexible interface residues have higher FoldX ΔΔG prediction error than mutations at rigid residues.*

---

## Project Structure

```
prot_test/
├── config.py              # All paths and parameters — edit before running
├── pipeline.py            # Main orchestrator: SKEMPI → ANM → FoldX → analysis
├── skempi.py              # Load and filter SKEMPI 2.0 dataset
├── structures.py          # Fetch PDB structures from RCSB
├── flexibility.py         # ANM flexibility scoring via ProDy
├── foldx.py               # FoldX RepairPDB + BuildModel wrappers
├── analysis.py            # Statistics, per-structure plots
├── features.py            # Feature extraction (mutation + structural + protein-level)
├── find_candidates.py     # Screen SKEMPI for best candidate complexes
├── ML/
│   └── regressor.py       # Random Forest regressor + classifier with LOPO CV
├── data/
│   ├── skempi_v2.csv      # Download from life.bsc.es/pid/skempi2
│   └── structures/        # PDB files (auto-fetched)
└── results/
    ├── <PDB_ID>.csv        # Per-structure results (used for resume)
    ├── mutations_combined.csv
    ├── feature_matrix_<target>.parquet
    ├── feature_importances_<target>.csv
    ├── lopo_cv_results_<target>.csv
    └── figures/
```

---

## Setup

### 1. Python environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. FoldX binary

Register for a free academic license at https://foldxsuite.crg.eu  
Download the binary, then update `config.py`:

```python
FOLDX_BIN = Path(r"C:\path\to\foldx_XXXXXXXX.exe")
```

### 3. SKEMPI 2.0 dataset

Download `skempi_v2.csv` from https://life.bsc.es/pid/skempi2  
Place it at `data/skempi_v2.csv`.

---

## Phase 1: Flexibility vs. FoldX Error

### Find candidate structures

```bash
python find_candidates.py
```

Scores SKEMPI complexes by mutation count, DDG spread, and mean B-factor.  
Paste the suggested `PILOT_PDB_IDS` into `config.py`.

### Run the pipeline

```bash
python pipeline.py
```

For each complex in `PILOT_PDB_IDS`:
1. Filters SKEMPI for single-point mutations at the interface
2. Downloads PDB structure from RCSB
3. Runs ANM (ProDy) → per-residue MSF flexibility scores (z-scored within chain)
4. Runs FoldX RepairPDB + BuildModel → predicted ΔΔG
5. Calibrates FoldX globally against experimental values → prediction error
6. Correlates flexibility with error (Spearman + Mann-Whitney)
7. Saves `results/<PDB_ID>.csv` and per-structure figures

**Resume behaviour:** if `results/<PDB_ID>.csv` exists, that structure is skipped.  
To force rerun: `del results\<PDB_ID>.csv`

### Key config options (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `PILOT_PDB_IDS` | `["1A22", ...]` | Structures to process. `None` = all SKEMPI |
| `RESOLUTION_CUTOFF` | `4.0` Å | Max crystal resolution |
| `INTERFACE_CUTOFF` | `8.0` Å | Cα distance to partner for interface definition |
| `ANM_MODES` | `10` | Number of slowest modes for MSF |
| `SKIP_FOLDX` | `False` | Use `\|DDG_exp\|` as target instead of FoldX error |

---

## Phase 2: Machine Learning

### Run regressor + classifier

```bash
cd ML
python regressor.py --target prediction_error
python regressor.py --target DDG
python regressor.py --target DDG --include-foldx
```

**Validation:** Leave-One-Protein-Out (LOPO) CV — trains on all structures except one, tests on held-out. Tests whether the model generalises to unseen proteins.

**Features used (34 total):**

| Category | Features |
|---|---|
| Flexibility | `msf_z`, `msf_z_neighbors_2/4`, `abs_msf_z`, `abs_msf_z_neighbors_2/4` |
| Mutation (signed) | `volume_change`, `hydrophobicity_change`, `charge_change`, `blosum62` |
| Mutation (absolute) | `abs_volume_change`, `abs_hydrophobicity_change`, `abs_charge_change`, `abs_blosum62` |
| Mutation flags | `is_to_gly/pro/ala/cys`, `is_from_gly/pro` |
| Interface location | `location_INT/COR/RIM/SUR/SUP` |
| Residue structural | `b_factor`, `dist_to_interface` |
| Protein-level | `prot_n_chains`, `prot_n_residues`, `prot_mean/std/max_bfactor`, `prot_n_interface_residues`, `prot_frac_interface` |

**Classification thresholds (prediction_error):**
- `< 0.5 kcal/mol` → accurate
- `0.5–1.5 kcal/mol` → moderate_error  
- `> 1.5 kcal/mol` → large_error

**Classification thresholds (DDG):**
- `< -0.5 kcal/mol` → stabilising
- `-0.5 to +0.5 kcal/mol` → neutral
- `> +0.5 kcal/mol` → destabilising

### Outputs

| File | Contents |
|---|---|
| `results/feature_matrix_<target>.parquet` | Full feature matrix for inspection |
| `results/feature_importances_<target>.csv` | Permutation + impurity importances |
| `results/lopo_cv_results_<target>.csv` | Per-fold LOPO regression results |
| `results/per_structure_cv_<target>.csv` | Within-protein CV results |
| `results/figures/feature_importances_<target>.png` | Importance bar charts |
| `results/figures/lopo_cv_<target>.png` | LOPO R² per protein |
| `results/figures/lopo_predictions_<target>.png` | Predicted vs actual scatter |
| `results/figures/confusion_clf_<target>.png` | Confusion matrices |

---

## Key Findings (pilot: 8 structures, 937 mutations)

**Top features for predicting FoldX error (permutation importance):**
1. `msf_z` (0.215) — ANM flexibility at mutated residue
2. `b_factor` (0.124) — crystallographic flexibility
3. `prot_frac_interface` (0.092) — fraction of protein at interface
4. `prot_mean_bfactor` (0.062) — overall protein flexibility

**LOPO CV:** mean R² ≈ −1.1 — model does not generalise across proteins  
**Per-structure CV:** R² = 0.15–0.47 for most structures — real signal within each protein

The gap between LOPO and per-structure performance indicates that the flexibility–error relationship is partly protein-family-specific. The top two features (`msf_z` and `b_factor`) consistently support the flexibility hypothesis within proteins, but different protein families require different calibrations.

---

## Dependencies

| Package | Purpose |
|---|---|
| `prody` | ANM normal mode analysis |
| `biopython` | PDB parsing, structure manipulation |
| `pandas`, `numpy`, `scipy` | Data and statistics |
| `scikit-learn` | Random Forest, Logistic Regression, CV |
| `matplotlib`, `seaborn` | Plotting |
| `pyarrow` | Parquet file I/O |
| FoldX binary | ΔΔG prediction (external, free academic) |

---

## Citation

If using SKEMPI 2.0:  
Jankauskaite et al. (2019) *Bioinformatics* 35(3):462–469. https://doi.org/10.1093/bioinformatics/bty635
