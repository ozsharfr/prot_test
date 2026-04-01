# Beyond the Static Structure
## Positional Context and Conformational Flexibility in Protein-Protein Binding Prediction

A computational pipeline for studying the role of **structural flexibility and positional context in ΔΔG prediction** — whether flexibility at interface residues explains binding effects and FoldX prediction error, and whether positional knowledge enables computational mutational scanning.

---

## Scientific Background

Standard binding affinity predictors (FoldX, Rosetta) assume a single rigid structure. Flexible regions at protein-protein interfaces may violate this assumption in ways that systematically bias predictions. This project quantifies:

1. How much structural flexibility explains FoldX prediction error
2. How well experimental binding effects (DDG) can be classified from biophysical features
3. Whether positional context enables computational prediction of untested substitutions at a site

**Features used:**
- **ANM (Anisotropic Normal Mode Analysis)** — per-residue MSF and ±2/4 neighbourhood averages
- **Crystallographic B-factors** — experimental flexibility proxy at residue and protein level
- **Physicochemical mutation properties** — volume, hydrophobicity, charge, BLOSUM62 (signed and absolute)
- **Interface location** — INT/COR/RIM/SUR/SUP from SKEMPI
- **Protein-level structural features** — chain count, residue count, interface size and fraction

**Data:** SKEMPI 2.0 — curated database of protein-protein binding mutations with experimental ΔΔG.

---

## Key Finding

> **Positional context is the dominant predictor of both binding effect and FoldX error.** Once some information is available about a residue position — from a few experimental measurements or from structural features — additional substitutions at that site can be screened computationally with meaningful accuracy. Flexibility features, particularly neighbourhood MSF, encode much of this positional context.

See `RESULTS.md` for full results and interpretation.

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
├── analysis.py            # Statistics and per-structure plots
├── features.py            # Feature extraction (mutation + structural + protein-level)
├── find_candidates.py     # Screen SKEMPI for best candidate complexes
├── ML/
│   ├── common.py          # Shared: data loading, CV, plotting, feature importance
│   ├── regressor.py       # RF + Ridge regressor with LOPO and per-structure CV
│   └── classifier.py      # RF classifier with LOPO and per-structure CV
├── data/
│   ├── skempi_v2.csv      # Download from life.bsc.es/pid/skempi2
│   └── structures/        # PDB files (auto-fetched by pipeline)
└── results/
    ├── <PDB_ID>.csv                      # Per-structure results (resume-safe)
    ├── feature_matrix_<target>.parquet   # Feature matrix for inspection/reuse
    ├── feature_importances_<target>.csv
    ├── lopo_cv_<target>.csv
    ├── per_structure_cv_<target>.csv
    └── figures/
```

---

## Setup

### 1. Python environment

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. FoldX binary

Register for a free academic license at https://foldxsuite.crg.eu and download the binary. Update `config.py`:

```python
FOLDX_BIN = Path(r"C:\path\to\foldx_XXXXXXXX.exe")
```

### 3. SKEMPI 2.0 dataset

Download `skempi_v2.csv` from https://life.bsc.es/pid/skempi2 and place at `data/skempi_v2.csv`.

---

## Phase 1 — Pipeline

### Find candidate structures

```cmd
python find_candidates.py
```

Scores SKEMPI complexes by mutation count, DDG spread, and mean B-factor. Paste suggested `PILOT_PDB_IDS` into `config.py`.

### Run the pipeline

```cmd
python pipeline.py
```

For each complex in `PILOT_PDB_IDS`:

1. Filters SKEMPI for single-point mutations at the interface
2. Downloads PDB structure from RCSB
3. Runs ANM (ProDy) → per-residue MSF z-scores + ±2/4 neighbourhood averages
4. Runs FoldX RepairPDB + BuildModel → predicted ΔΔG
5. Calibrates predictions globally → `prediction_error = |calibrated − experimental|`
6. Saves `results/<PDB_ID>.csv` — skips completed structures on rerun

**To force rerun a structure:** `del results\<PDB_ID>.csv`

### Key config options

| Parameter | Default | Description |
|---|---|---|
| `PILOT_PDB_IDS` | `["1A22", ...]` | Structures to process. `None` = all SKEMPI |
| `RESOLUTION_CUTOFF` | `4.0` Å | Maximum crystal resolution |
| `INTERFACE_CUTOFF` | `8.0` Å | Cα distance to partner chain for interface definition |
| `ANM_MODES` | `10` | Number of slowest normal modes for MSF |
| `SKIP_FOLDX` | `False` | Skip FoldX, use experimental DDG as target only |

---

## Phase 2 — Machine Learning

### Run the models

```cmd
cd ML
python regressor.py --target prediction_error
python regressor.py --target DDG
python classifier.py --target prediction_error
python classifier.py --target DDG
```

### Validation strategy

Three evaluations are reported for every model:

**1. Leave-One-Protein-Out (LOPO) CV**
Trains on all structures except one, tests on the held-out protein. The strictest test — measures generalisation to unseen protein families.

**2. Naive per-structure KFold CV** *(leakage baseline)*
Within each protein, mutations are randomly split across folds. Inflated by same-position leakage.

**3. Position-grouped per-structure CV** *(main within-protein evaluation)*
Within each protein, mutations at the same residue position are always kept in the same fold (GroupKFold by `resnum`). The gap between naive and position-grouped accuracy quantifies the value of positional information.

### Features (34 total)

| Category | Features |
|---|---|
| Flexibility (ANM) | `msf_z`, `msf_z_neighbors_2/4`, `abs_msf_z`, `abs_msf_z_neighbors_2/4` |
| Mutation — signed | `volume_change`, `hydrophobicity_change`, `charge_change`, `blosum62` |
| Mutation — absolute | `abs_volume_change`, `abs_hydrophobicity_change`, `abs_charge_change`, `abs_blosum62` |
| Mutation flags | `is_to_gly/pro/ala/cys`, `is_from_gly/pro` |
| Interface location | `location_INT/COR/RIM/SUR/SUP` |
| Residue structural | `b_factor`, `dist_to_interface` |
| Protein-level | `prot_n_chains`, `prot_n_residues`, `prot_mean/std/max_bfactor`, `prot_n_interface_residues`, `prot_frac_interface` |

### Classification thresholds

**`prediction_error`:** accurate < 0.5, moderate_error 0.5–1.5, large_error > 1.5 kcal/mol

**`DDG`:** stabilising < −0.5, neutral −0.5 to +0.5, destabilising > +0.5 kcal/mol

### Output files

| File | Contents |
|---|---|
| `results/feature_matrix_<target>.parquet` | Full feature matrix (X + target + pdb_id) |
| `results/feature_importances_<target>.csv` | Permutation + impurity importances |
| `results/lopo_cv_<target>.csv` | Per-fold LOPO results |
| `results/per_structure_cv_<target>.csv` | Position-grouped per-structure CV results |
| `results/figures/` | All plots |

---

## Dependencies

| Package | Purpose |
|---|---|
| `prody` | ANM normal mode analysis |
| `biopython` | PDB parsing and structure manipulation |
| `pandas`, `numpy`, `scipy` | Data processing and statistics |
| `scikit-learn` | Random Forest, Ridge, GroupKFold CV, permutation importance |
| `matplotlib` | Plotting |
| `pyarrow` | Parquet file I/O |
| FoldX binary | ΔΔG prediction (external, free academic license) |

---

## Citation

If using SKEMPI 2.0:
Jankauskaite et al. (2019) *Bioinformatics* 35(3):462–469. https://doi.org/10.1093/bioinformatics/bty635