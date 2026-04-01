"""
regressor.py — Predict DDG target from biophysical features using Random Forest.

Validation: Leave-One-Protein-Out (LOPO) CV + per-structure GroupKFold CV.
Run: python regressor.py --target prediction_error
     python regressor.py --target DDG
     python regressor.py --target DDG --include-foldx
"""

from common import (
    _PROJECT_ROOT, log, np, pd, plt,
    RESULTS_DIR, FIGURES_DIR, DEFAULT_TARGET,
    load_results, prepare_xy, build_pdb_paths,
    lopo_cv, per_structure_cv,
    get_feature_importances, plot_importances, plot_per_structure_results,
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------------------------
# Regression-specific CV wrappers
# ---------------------------------------------------------------------------

def _r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_lopo(model, X, y, groups, name):
    return lopo_cv(model, X, y, groups, name, scorer=_r2, score_name="r2")


def run_per_structure(model, X, y, groups, name, pos_groups=None):
    return per_structure_cv(model, X, y, groups, name,
                            pos_groups=pos_groups,
                            scorer=_r2, score_name="r2")


# ---------------------------------------------------------------------------
# Regression-specific plots
# ---------------------------------------------------------------------------

def plot_lopo_results(all_cv: dict, title: str, out_path):
    models     = list(all_cv.keys())
    proteins   = sorted(set(fold for df in all_cv.values() for fold in df["fold"].tolist()))
    x          = np.arange(len(proteins))
    width      = 0.8 / max(len(models), 1)
    colors     = ["#534AB7", "#1D9E75", "#D85A30"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, (mname, df) in enumerate(all_cv.items()):
        r2_vals = [df.loc[df["fold"] == p, "r2"].values[0]
                   if p in df["fold"].values else np.nan for p in proteins]
        ax.bar(x + i * width, r2_vals, width, label=mname,
               color=colors[i % len(colors)], alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(proteins, rotation=30, ha="right")
    ax.set_ylabel("R² (held-out protein)")
    ax.set_title("LOPO CV — R² per held-out structure")
    ax.legend()

    ax = axes[1]
    means = [df["r2"].mean() for df in all_cv.values()]
    stds  = [df["r2"].std()  for df in all_cv.values()]
    ax.bar(models, means, yerr=stds, color=colors[:len(models)], alpha=0.8, capsize=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Mean R²")
    ax.set_title("LOPO CV — mean R² by model")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", out_path)
    plt.close()


def plot_lopo_predictions(model, X, y, groups, target, out_path):
    proteins   = groups.unique()
    y_pred_all = pd.Series(index=y.index, dtype=float)

    for held_out in proteins:
        train_mask = groups != held_out
        test_mask  = groups == held_out
        if train_mask.sum() < 5:
            continue
        model.fit(X[train_mask], y[train_mask])
        y_pred_all[test_mask] = model.predict(X[test_mask])

    valid = y_pred_all.dropna()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for prot in proteins:
        mask = (groups == prot) & y_pred_all.notna()
        ax.scatter(y[mask], y_pred_all[mask], alpha=0.5, s=20, label=prot)

    lim = [min(y.min(), valid.min()) - 0.1, max(y.max(), valid.max()) + 0.1]
    ax.plot(lim, lim, "k--", linewidth=1)
    r2 = r2_score(y[y_pred_all.notna()], valid)
    ax.set_xlabel(f"Actual {target} (kcal/mol)")
    ax.set_ylabel(f"LOPO predicted {target} (kcal/mol)")
    ax.set_title(f"LOPO predictions (R²={r2:.3f})\nColour = held-out protein")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info("Saved: %s", out_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Random Forest regressor for DDG prediction")
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

    # Save feature matrix
    feature_df = X.copy()
    feature_df[args.target] = y.values
    feature_df["pdb_id"]    = groups.values
    out_parquet = RESULTS_DIR / f"feature_matrix_{args.target}.parquet"
    feature_df.to_parquet(out_parquet, index=False)
    log.info("Feature matrix saved: %s (%d rows × %d features)",
             out_parquet, len(feature_df), len(feature_names))

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    models = {
        # Conservative RF — shallow trees, many samples per leaf to prevent overfit
        "RF (conservative)": Pipeline([
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=3,
                                         min_samples_leaf=5,
                                         random_state=42, n_jobs=-1))
        ]),
        # Ridge — linear model, much less prone to overfit with few positions
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0))
        ]),
    }
    # Keep a reference to RF for feature importances
    rf = models["RF (conservative)"]

    # LOPO CV
    print(f"\n{'='*65}")
    print(f"Target: {args.target} | LOPO CV | n={len(X)} | structures={groups.nunique()}")
    print(f"{'='*65}")

    all_cv = {name: run_lopo(model, X, y, groups, name) for name, model in models.items()}

    print(f"\n{'Model':<22} {'Mean R²':>9} {'Std R²':>8} {'Mean RMSE':>10}")
    print(f"{'-'*55}")
    for name, cv_df in all_cv.items():
        if cv_df.empty:
            continue
        rmse_mean = cv_df["rmse"].mean() if "rmse" in cv_df.columns else float("nan")
        print(f"{name:<22} {cv_df['r2'].mean():>9.3f} {cv_df['r2'].std():>8.3f} "
              f"{rmse_mean:>10.3f}")
    print(f"{'='*55}")

    # Per-structure CV — naive (random split, may leak same-position mutations)
    print(f"\n{'='*65}")
    print("Per-structure CV — naive KFold (no position grouping)")
    print(f"{'='*65}")
    per_struct_naive = {name: run_per_structure(model, X, y, groups, name, pos_groups=None)
                        for name, model in models.items()}
    for name, df in per_struct_naive.items():
        if df.empty:
            continue
        valid = df.dropna(subset=["r2_mean"])
        if valid.empty:
            continue
        weighted_avg = np.average(valid["r2_mean"].values, weights=valid["n"].values)
        print(f"  {name} — weighted avg R²: {weighted_avg:.3f}  (naive, leaky)")

    # Per-structure CV — position-grouped (no leakage)
    print(f"\n{'='*65}")
    print("Per-structure CV (position-grouped, no leakage)")
    print(f"{'='*65}")

    pos_groups = df_valid["resnum"] if "resnum" in df_valid.columns else None
    per_struct = {name: run_per_structure(model, X, y, groups, name, pos_groups)
                  for name, model in models.items()}

    all_structures = sorted(set(
        s for df in per_struct.values() for s in df["structure"].tolist()
    ))
    print(f"\n{'Structure':<12} {'Model':<22} {'N':>5} {'Pos':>5} {'R²':>8} {'±':>6} {'RMSE':>8}")
    print(f"{'-'*70}")
    for prot in all_structures:
        for name, df in per_struct.items():
            row = df[df["structure"] == prot]
            if row.empty:
                continue
            r     = row.iloc[0]
            n_pos = int(r["n_positions"]) if "n_positions" in r.index else "?"
            r2   = f"{r['r2_mean']:>8.3f}" if not pd.isna(r['r2_mean']) else "     nan"
            rmse = f"{r['rmse_mean']:>8.3f}" if not pd.isna(r['rmse_mean']) else "     nan"
            print(f"{prot:<12} {name:<22} {int(r['n']):>5} {str(n_pos):>5} "
                  f"{r2} {r['r2_std']:>6.3f} {rmse}")
        print()

    # Per-structure summary statistics
    print(f"\n{'='*65}")
    print("Per-structure summary")
    print(f"{'='*65}")
    for name, df in per_struct.items():
        if df.empty:
            continue
        valid = df.dropna(subset=["r2_mean"])
        if valid.empty:
            continue
        struct_avg   = valid["r2_mean"].mean()
        struct_std   = valid["r2_mean"].std()
        weights      = valid["n"].values
        weighted_avg = np.average(valid["r2_mean"].values, weights=weights)
        print(f"\n{name}:")
        print(f"  Structures-average R²  : {struct_avg:.3f} ± {struct_std:.3f}  "
              f"(n_structures={len(valid)})")
        print(f"  Weighted average R²    : {weighted_avg:.3f}  "
              f"(weighted by n_mutations)")
        print(f"  Structures with R² > 0 : "
              f"{(valid['r2_mean'] > 0).sum()}/{len(valid)}")

    # Feature importances
    imp_df = get_feature_importances(rf, X, y, feature_names)
    print(f"\nTop features (permutation importance):")
    print(imp_df[["feature", "perm_importance", "perm_importance_std"]].head(10).to_string(index=False))

    # Plots
    plot_lopo_results(all_cv, f"LOPO CV — {args.target}",
                      FIGURES_DIR / f"lopo_cv_{args.target}.png")
    plot_importances(imp_df, f"Feature importances — {args.target}",
                     FIGURES_DIR / f"feature_importances_{args.target}.png")
    plot_lopo_predictions(rf, X, y, groups, args.target,
                          FIGURES_DIR / f"lopo_predictions_{args.target}.png")
    plot_per_structure_results(per_struct, "r2", f"Per-structure CV — {args.target}",
                               FIGURES_DIR / f"per_structure_cv_{args.target}.png")

    # Save CSVs
    imp_df.to_csv(RESULTS_DIR / f"feature_importances_{args.target}.csv", index=False)
    pd.concat([df.assign(model=n) for n, df in all_cv.items()], ignore_index=True
              ).to_csv(RESULTS_DIR / f"lopo_cv_{args.target}.csv", index=False)
    pd.concat([df.assign(model=n) for n, df in per_struct.items()], ignore_index=True
              ).to_csv(RESULTS_DIR / f"per_structure_cv_{args.target}.csv", index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()