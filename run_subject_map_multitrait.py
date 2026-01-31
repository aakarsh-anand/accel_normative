# run_subject_map_multitrait.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.analysis.viz import savefig


def build_primary_trait_assignment(
    trait_names: list[str],
    traits_root: Path,
    ids_all: np.ndarray,
    id_col: str = "Participant ID",
) -> pd.DataFrame:
    ids_all = ids_all.astype(str)
    base = pd.DataFrame({id_col: ids_all}).set_index(id_col)

    best_trait = pd.Series(index=base.index, dtype="object")
    best_tte = pd.Series(index=base.index, dtype="float64")
    n_pos = pd.Series(0, index=base.index, dtype="int64")

    for name in trait_names:
        lab_path = traits_root / name / "labels.csv"
        if not lab_path.exists():
            continue

        lab = pd.read_csv(lab_path)
        if id_col not in lab.columns or "Label" not in lab.columns or "TTE_years" not in lab.columns:
            continue

        lab[id_col] = lab[id_col].astype(str)
        lab = lab.set_index(id_col)

        # positives only
        pos = lab[lab["Label"] == 1][["TTE_years"]]
        pos = pos.reindex(base.index)

        tte = pos["TTE_years"]
        is_pos = tte.notna()

        # count how many traits each person is positive for
        n_pos.loc[is_pos] += 1

        # initialize best_tte
        need_init = is_pos & best_tte.isna()
        best_tte.loc[need_init] = tte.loc[need_init]
        best_trait.loc[need_init] = name

        # update if earlier event (smaller TTE)
        better = is_pos & best_tte.notna() & (tte < best_tte)
        best_tte.loc[better] = tte.loc[better]
        best_trait.loc[better] = name

    out = base.copy()
    out["primary_trait"] = best_trait.fillna("Control")
    out["primary_tte"] = best_tte
    out["n_traits_positive"] = n_pos
    out = out.reset_index()
    return out


def make_palette(labels: list[str]):
    # Build a palette that can handle >20 categories by cycling tab20 variants
    cmaps = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
    colors = {}
    k = 0
    for lab in labels:
        cmap = cmaps[(k // 20) % len(cmaps)]
        colors[lab] = cmap(k % 20)
        k += 1
    colors["Control"] = (0.6, 0.6, 0.6, 0.20)  # light gray transparent
    colors["Other"] = (0.2, 0.2, 0.2, 0.6)
    return colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resid_dir", required=True)
    ap.add_argument("--traits_yaml", default="configs/traits.yaml")
    ap.add_argument("--traits_root", default="outputs/traits")
    ap.add_argument("--outdir", default="outputs/subject_maps_multitrait")
    ap.add_argument("--resid_key", default="resid")
    ap.add_argument("--id_col", default="Participant ID")
    ap.add_argument("--max_controls", type=int, default=30000)
    ap.add_argument("--top_k_traits", type=int, default=20, help="Show only top K traits; rest -> Other")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load residuals + ids
    npz = np.load(Path(args.resid_dir) / "normative_residuals.npz")
    if args.resid_key not in npz.files:
        raise KeyError(f"{args.resid_key} not in residual npz; available: {npz.files}")
    X = npz[args.resid_key].astype(np.float32)

    summ = pd.read_csv(Path(args.resid_dir) / "residual_summary.csv")
    ids = summ[args.id_col].astype(str).values

    # Read trait list
    with open(args.traits_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    trait_names = [t for t in cfg.get("traits", [])]

    traits_root = Path(args.traits_root)

    # Build primary trait assignment
    assign = build_primary_trait_assignment(
        trait_names=trait_names,
        traits_root=traits_root,
        ids_all=ids,
        id_col=args.id_col,
    )

    # Reduce label space: keep only top_k traits by count, else Other (except Control)
    counts = assign["primary_trait"].value_counts()
    top_traits = [t for t in counts.index if t != "Control"][: args.top_k_traits]
    assign["plot_trait"] = assign["primary_trait"].where(
        assign["primary_trait"].isin(top_traits) | (assign["primary_trait"] == "Control"),
        "Other",
    )

    # Subsample controls for plotting / fitting
    rng = np.random.default_rng(args.seed)
    is_control = assign["plot_trait"].values == "Control"
    idx_control = np.where(is_control)[0]
    idx_noncontrol = np.where(~is_control)[0]

    if len(idx_control) > args.max_controls:
        idx_control = rng.choice(idx_control, size=args.max_controls, replace=False)

    idx_plot = np.sort(np.concatenate([idx_control, idx_noncontrol]))
    X_plot = X[idx_plot]
    assign_plot = assign.iloc[idx_plot].reset_index(drop=True)

    # Fit a single global UMAP on plot subset
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=args.seed, metric="euclidean")
    emb2 = reducer.fit_transform(X_plot)

    # Plot
    labels = list(pd.unique(assign_plot["plot_trait"].astype(str)))
    # Ensure control first for background
    labels = ["Control"] + [l for l in labels if l not in ("Control",)]
    pal = make_palette([l for l in labels if l not in ("Control", "Other")] + ["Other"])

    plt.figure(figsize=(8.0, 6.8))

    # background controls
    m = assign_plot["plot_trait"].astype(str).values == "Control"
    plt.scatter(emb2[m, 0], emb2[m, 1], s=2, alpha=0.08, label="Control")

    # other categories
    for lab in [l for l in labels if l != "Control"]:
        m = assign_plot["plot_trait"].astype(str).values == lab
        if not m.any():
            continue
        plt.scatter(emb2[m, 0], emb2[m, 1], s=10, alpha=0.75, label=lab, color=pal.get(lab, None))

    plt.title(f"Subject UMAP colored by primary trait (top {args.top_k_traits} + Other)")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(markerscale=2, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(outdir / "subjects_umap_multitrait.png")

    assign.to_csv(outdir / "primary_trait_assignment.csv", index=False)
    print("[OK] wrote:", outdir)


if __name__ == "__main__":
    main()
