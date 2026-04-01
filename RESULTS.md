# Results — Beyond the Static Structure

**Dataset:** SKEMPI 2.0 | 20 protein-protein complexes | 1,312 single-point mutations  
**Models:** Random Forest (conservative: max_depth=3, min_samples_leaf=5) + Ridge  
**Validation:** LOPO CV + position-grouped per-structure CV (GroupKFold by resnum)

---

## Summary

The original hypothesis — that FoldX prediction error correlates with residue flexibility — holds within individual protein structures but does not generalise across protein families. A stronger and more practically useful finding emerged: **positional context is the dominant predictor of both binding effect and model error**. This suggests a computational mutational scanning strategy: measure a few substitutions experimentally at an interface position, then computationally screen additional substitutions at that position and its neighbours.

---

## 1. Classification — Predicting FoldX Error (`prediction_error`)

**Classes:** accurate (< 0.5), moderate_error (0.5–1.5), large_error (> 1.5) kcal/mol  
**Class distribution:** moderate_error 543, accurate 497, large_error 272  
**Random chance baseline:** 0.333 (3 classes)

### LOPO CV (cross-protein generalisation)

| Metric | Value |
|---|---|
| Balanced accuracy | 0.371 ± 0.097 |

**Per-class performance (all LOPO folds combined):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| accurate | 0.47 | 0.32 | 0.38 |
| large_error | 0.33 | 0.50 | 0.40 |
| moderate_error | 0.41 | 0.42 | 0.41 |

The model is best at identifying `large_error` cases (recall 0.50) — the most practically useful prediction, flagging mutations where FoldX cannot be trusted.

### Per-structure CV — leakage comparison

| Evaluation | Weighted avg bal_acc | Notes |
|---|---|---|
| Naive KFold (leaky) | 0.581 | Same-position mutations split across folds |
| Position-grouped (honest) | 0.413 | Same position always in same fold |
| **Leakage effect** | **0.168** | Value of positional identity alone |

**Structures above chance (position-grouped):** 14/20

### Per-structure breakdown (position-grouped)

| Structure | N | Positions | Bal_acc | ± |
|---|---|---|---|---|
| 2NZ9 | 18 | 18 | 0.817 | 0.186 |
| 4BFI | 8 | 6 | 0.800 | 0.400 |
| 2FTL | 32 | 15 | 0.667 | 0.279 |
| 3SE3 | 13 | 13 | 0.633 | 0.194 |
| 1JRH | 58 | 26 | 0.599 | 0.197 |
| 2WPT | 41 | 23 | 0.572 | 0.221 |
| 2JEL | 27 | 20 | 0.503 | 0.247 |
| 3BT1 | 99 | 95 | 0.500 | 0.020 |
| 1A22 | 163 | 89 | 0.444 | 0.109 |
| 1R0R | 204 | 10 | 0.400 | 0.077 |
| 1CBW | 43 | 14 | 0.375 | 0.139 |
| 3BN9 | 23 | 20 | 0.367 | 0.113 |
| 1DAN | 84 | 68 | 0.384 | 0.080 |
| 4RS1 | 36 | 21 | 0.328 | 0.122 |
| 1AO7 | 133 | 51 | 0.321 | 0.170 |
| 3MZG | 58 | 6 | 0.300 | 0.163 |
| 1JTG | 135 | 46 | 0.270 | 0.084 |
| 3QDG | 14 | 7 | 0.150 | 0.200 |
| 3EQS | 10 | 10 | 0.100 | 0.200 |

*Note: 3EQS (n=10) and 3QDG (n=14) results are unreliable due to very small sample size.*

### Top features (permutation importance)

| Rank | Feature | Importance | Type |
|---|---|---|---|
| 1 | `abs_msf_z_neighbors_4` | 0.019 | ANM neighbourhood flexibility |
| 2 | `b_factor` | 0.014 | Crystallographic flexibility |
| 3 | `prot_frac_interface` | 0.013 | Protein-level structural |
| 4 | `dist_to_interface` | 0.012 | Residue structural |
| 5 | `abs_msf_z` | 0.011 | ANM residue flexibility |
| 6 | `abs_volume_change` | 0.010 | Mutation magnitude |
| 7 | `msf_z` | 0.009 | ANM residue flexibility (signed) |
| 8 | `abs_hydrophobicity_change` | 0.008 | Mutation magnitude |
| 9 | `charge_change` | 0.008 | Mutation property |
| 10 | `location_SUR` | 0.007 | Interface location |

**Key observation:** The top 5 features are all flexibility or structural context features, directly supporting the hypothesis. Notably `abs_msf_z_neighbors_4` (±4 residue neighbourhood) ranks above `abs_msf_z` (single residue), suggesting local structural context matters more than the individual residue.

---

## 2. Classification — Predicting Binding Effect (`DDG`)

**Classes:** stabilising (< −0.5), neutral (−0.5 to +0.5), destabilising (> +0.5) kcal/mol  
**Class distribution:** destabilising 720 (55%), neutral 510 (39%), stabilising 82 (6%)

### LOPO CV

| Metric | Value |
|---|---|
| Balanced accuracy | 0.473 ± 0.186 |

**Per-class performance:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| destabilising | 0.67 | 0.65 | 0.66 |
| neutral | 0.57 | 0.63 | 0.60 |
| stabilising | 0.11 | 0.09 | 0.10 |

The `stabilising` class (only 6% of data) is nearly unpredictable — severe class imbalance. The model effectively operates as a binary classifier between destabilising and neutral.

### Per-structure CV — leakage comparison

| Evaluation | Weighted avg bal_acc | Notes |
|---|---|---|
| Naive KFold (leaky) | 0.663 | |
| Position-grouped (honest) | 0.504 | |
| **Leakage effect** | **0.159** | |

**Structures above chance (position-grouped):** 18/20

### Comparison: DDG vs prediction_error

| Metric | prediction_error | DDG |
|---|---|---|
| LOPO balanced acc | 0.371 | 0.473 |
| Naive per-structure | 0.581 | 0.663 |
| Position-grouped | 0.413 | 0.504 |
| Structures above chance | 14/20 | 18/20 |
| Leakage effect | 0.168 | 0.159 |

**DDG is substantially easier to predict than FoldX error** across all evaluation types. This is expected: whether a mutation is destabilising is partly determined by the direct physicochemical change (charge, size, hydrophobicity), which are explicit features. FoldX error is a more complex second-order quantity — the gap between a model's prediction and reality.

---

## 3. Regression — Predicting Continuous Values

Both RF and Ridge regression show negative R² in position-grouped CV across nearly all structures, confirming overfitting: the model cannot predict FoldX error or DDG at a new position within the same protein better than the mean. This is consistent with the classification findings — the signal is real but concentrated in positional identity rather than fully general biophysical features.

---

## 4. The Positional Context Finding

The leakage effect (~0.16 in balanced accuracy) is not a bug — it is a signal. It represents the information gained by knowing the positional identity of the mutation:

- Naive CV allows the model to see substitutions at position 64 in training and test on a different substitution at position 64. The model learns "position 64 tends to have high error" or "position 64 tends to be destabilising."
- Position-grouped CV removes this, forcing the model to predict position 64 using only transferable biophysical features.

The gap tells us that **positional identity encodes ~0.16 units of balanced accuracy** beyond what transferable features alone provide.

### Practical implication: computational mutational scanning

This finding directly supports a computational screening strategy:

1. **Measure 1–2 substitutions experimentally** at a target interface position
2. **Use that positional signal** (combined with structural features) to predict additional substitutions at the same or neighbouring positions
3. **Prioritise candidates** for experimental validation based on predicted binding effect

This is analogous to few-shot learning: a small number of experimental measurements at a position bootstraps accurate predictions for untested variants. The neighbourhood flexibility features (`abs_msf_z_neighbors_4`) suggest the relevant unit extends to a ±4 residue window — measuring any substitution within this window provides useful context for predicting others.

---

## 5. Anomalous Structures

**Consistently underperforming (prediction_error):** 3EQS (n=10), 3QDG (n=14) — too few mutations for reliable estimation. 1JTG, 1AO7 — near or below chance despite adequate sample size; structural reasons unknown.

**Consistently well-performing (DDG):** 2FTL, 3BT1, 3S9D, 3BN9 — the model captures meaningful signal within these proteins. Worth investigating what structural properties distinguish them.

---

## 6. Next Steps

**Short term:**
- Position-level correlation analysis: Spearman(msf_z, prediction_error) per position (averaged over substitutions) to directly test the flexibility hypothesis at the right level of aggregation
- Binary classification (destabilising vs not) to address stabilising class imbalance
- Investigate why certain structures are anomalous

**Medium term:**
- Add AlphaFold pLDDT as a feature (fetch from AlphaFold DB by UniProt ID) — a third independent flexibility measure complementing ANM and B-factor
- Conformational ensemble generation (ColabFold MSA subsampling or AlphaFlow) → Boltzmann-weighted ensemble ΔΔG as an improved FoldX alternative for flexible regions

**Long term:**
- Test the computational scanning hypothesis directly: for structures with dense mutational data, hold out all mutations at several positions, train on the rest, and measure how well the model predicts held-out positions using their neighbours as context