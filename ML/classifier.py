"""
classifier.py — Classify mutations by DDG or prediction_error using Random Forest.

Validation: Leave-One-Protein-Out (LOPO) CV + per-structure GroupKFold CV.
Run: python classifier.py --target prediction_error
     python classifier.py --target DDG
"""

from common import (
    _PROJECT_ROOT, log, np, pd, plt,
    RESULTS_DIR, FIGURES_DIR, DEFAULT_TARGET,
    PREDICTION_ERROR_THRESHOLDS, DDG_THRESHOLDS,
    ERROR_LABELS, CLASS_LABELS,
    load_results, prepare_xy, build_pdb_paths,
    lopo_cv, per_structure_cv,
    get_feature_importances, plot_importances, plot_per_structure_results,
    binarize_target,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix)


# ---------------------------------------------------------------------------
# Classification-specific CV wrappers
# ---------------------------------------------------------------------------

def run_lopo_classify(model, X, y_cls, groups, name):
    return lopo_cv(model, X, y_cls, groups, name,
                   scorer=balanced_accuracy_score,
                   score_name="balanced_acc")


def run_per_structure_classify(model, X, y_cls, groups, name, pos_groups=None):
    return per_structure_cv(model, X, y_cls, groups, name,
                             pos_groups=pos_groups,
                             scorer=balanced_accuracy_score,
                             score_name="bal_acc")


# ---------------------------------------------------------------------------
# Classification-specific plots
# ---------------------------------------------------------------------------

def plot_confusion_matrices(model, X, y_cls, groups, labels, title, out_path):
    """One confusion matrix per held-out protein (LOPO predictions)."""
    proteins = groups.unique()
    n        = len(proteins)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, held_out in zip(axes, proteins):
        train_mask = groups != held_out
        test_mask  = groups == held_out
        if train_mask.sum() < 5:
            continue
        model.fit(X[train_mask], y_cls[train_mask])
        y_pred = model.predict(X[test_mask])
        cm = confusion_matrix(y_cls[test_mask], y_pred, labels=labels)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"held-out: {held_out}")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Random Forest classifier for DDG prediction")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        choices=["prediction_error", "DDG"])
    parser.add_argument("--include-foldx", action="store_true",
                        help="Add ddg_foldx as feature when target=DDG")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df        = load_results(RESULTS_DIR)
    pdb_paths = build_pdb_paths(df)

    X, y, groups, feature_names, df_valid = prepare_xy(
        df, target=args.target,
        include_foldx=args.include_foldx,
        pdb_paths=pdb_paths or None
    )

    # Binarize target
    thresholds = PREDICTION_ERROR_THRESHOLDS if args.target == "prediction_error" else DDG_THRESHOLDS
    labels     = ERROR_LABELS if args.target == "prediction_error" else CLASS_LABELS
    y_cls      = binarize_target(y, thresholds=thresholds, labels=labels)

    valid  = y_cls.notna()
    X_c    = X[valid]
    y_c    = y_cls[valid]
    g_c    = groups[valid]
    pos_c  = df_valid.loc[X_c.index, "resnum"] if "resnum" in df_valid.columns else None

    rf  = make_pipeline(RandomForestClassifier(n_estimators=200, max_depth=5,
                                               class_weight="balanced", random_state=42))
    models = {"RF Classifier": rf}

    # LOPO CV
    print(f"\n{'='*65}")
    print(f"Classification — target: {args.target} | LOPO CV | n={len(X_c)}")
    print(f"{'='*65}")

    all_lopo = {name: run_lopo_classify(model, X_c, y_c, g_c, name)
                for name, model in models.items()}

    for name, cv_df in all_lopo.items():
        if cv_df.empty:
            continue
        print(f"\n{name} — mean balanced acc: {cv_df['balanced_acc'].mean():.3f} "
              f"± {cv_df['balanced_acc'].std():.3f}")

    # Combined classification report across all LOPO folds
    print(f"\nLOPO classification report (all folds combined):")
    all_true, all_pred = [], []
    for held_out in g_c.unique():
        train_mask = g_c != held_out
        test_mask  = g_c == held_out
        if train_mask.sum() < 5:
            continue
        rf.fit(X_c[train_mask], y_c[train_mask])
        all_true.extend(y_c[test_mask].tolist())
        all_pred.extend(rf.predict(X_c[test_mask]).tolist())
    print(classification_report(all_true, all_pred, zero_division=0))

    # Per-structure CV — naive (may leak same-position mutations)
    print(f"\n{'='*65}")
    print("Per-structure classification — naive KFold (no position grouping)")
    print(f"{'='*65}")
    per_struct_naive = {name: run_per_structure_classify(model, X_c, y_c, g_c, name, pos_groups=None)
                        for name, model in models.items()}
    for name, df in per_struct_naive.items():
        if df.empty:
            continue
        valid = df.dropna(subset=["bal_acc_mean"])
        if valid.empty:
            continue
        weighted_avg = np.average(valid["bal_acc_mean"].values, weights=valid["n"].values)
        print(f"  {name} — weighted avg bal_acc: {weighted_avg:.3f}  (naive, leaky)")

    # Per-structure CV — position-grouped (no leakage)
    print(f"\n{'='*65}")
    print("Per-structure classification (position-grouped, no leakage)")
    print(f"{'='*65}")

    per_struct = {name: run_per_structure_classify(model, X_c, y_c, g_c, name, pos_c)
                  for name, model in models.items()}

    all_structures = sorted(set(
        s for df in per_struct.values() for s in df["structure"].tolist()
    ))
    print(f"\n{'Structure':<12} {'Model':<22} {'N':>5} {'Pos':>5} {'bal_acc':>9} {'±':>6}")
    print(f"{'-'*65}")
    for prot in all_structures:
        for name, df in per_struct.items():
            row = df[df["structure"] == prot]
            if row.empty:
                continue
            r     = row.iloc[0]
            n_pos = int(r["n_positions"]) if "n_positions" in r.index else "?"
            print(f"{prot:<12} {name:<22} {int(r['n']):>5} {str(n_pos):>5} "
                  f"{r['bal_acc_mean']:>9.3f} {r['bal_acc_std']:>6.3f}")
        print()

    # Per-structure summary statistics
    print(f"\n{'='*65}")
    print("Per-structure summary")
    print(f"{'='*65}")
    for name, df in per_struct.items():
        if df.empty:
            continue
        valid = df.dropna(subset=["bal_acc_mean"])
        if valid.empty:
            continue
        # Simple average across structures (each structure equally weighted)
        struct_avg = valid["bal_acc_mean"].mean()
        struct_std = valid["bal_acc_mean"].std()
        # Weighted average — each structure weighted by number of mutations
        weights     = valid["n"].values
        weighted_avg = np.average(valid["bal_acc_mean"].values, weights=weights)
        print(f"\n{name}:")
        print(f"  Structures-average bal_acc : {struct_avg:.3f} ± {struct_std:.3f}  "
              f"(n_structures={len(valid)})")
        print(f"  Weighted average bal_acc   : {weighted_avg:.3f}  "
              f"(weighted by n_mutations)")
        print(f"  Random chance              :  0.333  (3 classes)")
        print(f"  Structures above chance    : "
              f"{(valid['bal_acc_mean'] > 0.333).sum()}/{len(valid)}")

    # Feature importances
    imp_df = get_feature_importances(rf, X_c, y_c, feature_names)
    print(f"\nTop features (permutation importance):")
    print(imp_df[["feature", "perm_importance", "perm_importance_std"]].head(10).to_string(index=False))

    tag = f"clf_{args.target}"

    # Plots
    plot_confusion_matrices(rf, X_c, y_c, g_c, labels,
                            f"RF confusion matrices — {args.target}",
                            FIGURES_DIR / f"confusion_{tag}.png")
    plot_importances(imp_df, f"Feature importances (classifier) — {args.target}",
                     FIGURES_DIR / f"feature_importances_{tag}.png")
    plot_per_structure_results(per_struct, "bal_acc",
                               f"Per-structure classification — {args.target}",
                               FIGURES_DIR / f"per_structure_{tag}.png")

    # Save CSVs
    imp_df.to_csv(RESULTS_DIR / f"feature_importances_{tag}.csv", index=False)
    pd.concat([df.assign(model=n) for n, df in all_lopo.items()], ignore_index=True
              ).to_csv(RESULTS_DIR / f"lopo_{tag}.csv", index=False)
    pd.concat([df.assign(model=n) for n, df in per_struct.items()], ignore_index=True
              ).to_csv(RESULTS_DIR / f"per_structure_{tag}.csv", index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()