#!/usr/bin/env python3
"""
RQ3 - How many reviews are needed to trust a former-vs-current rating gap?

Implements Sections 3.1, 4.3 and 5.1 of the midterm report:

  * reference employers = employers with >= 1,000 former AND >= 1,000 current reviews;
    each one's full-data gap is treated as its true ("reference") gap
  * for n = 50, 60, ..., 300: draw n former and n current reviews with replacement from
    the same employer, compute the gap, repeat 1,000 times
  * n* = smallest n at which 95% of draws, pooled across reference employers, fall
    within +/-0.25 of the reference gap
  * baseline = normal approximation  n* = 2 * (1.96 * sigma / 0.25)^2   (183 in the report)
  * H3: more than half of employers do not have n* reviews in both groups
        (rejected if at least half of ALL employers have >= n* former and >= n* current)
  * sensitivity: tolerance 0.20 / 0.25 / 0.30 and confidence level 90% / 95%

Usage (run from the repo root; the three rq*.py files must sit in the same folder)
-----
    python rq3_reviews_needed.py --data data/processed/processed.parquet

In a notebook:
    import sys; sys.path.append("..")
    from rq3_reviews_needed import run_rq3, print_report
    from rq1_rating_gap import load_processed
    df, log = load_processed("../data/processed/processed.parquet")
    res = run_rq3(df); print_report(res)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_rating_gap import BAR, BLUE, GRID, INK, INK_2, ORANGE, SEED, SURFACE, _style, load_processed

# --------------------------------------------------------------------------- #
# Settings (from the report)
# --------------------------------------------------------------------------- #
REF_MIN_EACH = 1000                  # reference employers: >= 1,000 former and >= 1,000 current
N_GRID = list(range(50, 301, 10))    # n = 50 ... 300 per group
N_DRAWS = 1000                       # draws per (employer, n)
TOLERANCE = 0.25                     # main tolerance (half of the pooled 0.50 gap)
LEVEL = 0.95                         # main confidence level
TOLERANCES = [0.20, 0.25, 0.30]      # sensitivity
LEVELS = [0.90, 0.95]                # sensitivity
Z = 1.96


# --------------------------------------------------------------------------- #
# Baseline: normal approximation
# --------------------------------------------------------------------------- #
def normal_approx_n(sigma: float, tol: float = TOLERANCE) -> int:
    """n per group so that the 95% margin of a difference of two means is <= tol."""
    return int(np.ceil(2 * (Z * sigma / tol) ** 2))


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def reference_distributions(df: pd.DataFrame, min_each: int = REF_MIN_EACH) -> list[dict]:
    """
    For each reference employer, the empirical rating distribution of former and current reviews.
    Drawing n reviews with replacement from an employer's reviews is the same as drawing
    Multinomial(n, p) counts over its distinct rating values, which is much faster.
    """
    counts = df.groupby(["company", "former"], observed=True).size().unstack(fill_value=0)
    ref_ids = counts.index[(counts.get(True, 0) >= min_each) & (counts.get(False, 0) >= min_each)]
    tab = (df[df["company"].isin(ref_ids)]
           .groupby(["company", "former", "rating"], observed=True).size())
    out = []
    for firm in ref_ids:
        d = {"firm_id": firm}
        for status, key in ((True, "f"), (False, "c")):
            s = tab.loc[(firm, status)]
            s = s[s > 0]
            d[f"vals_{key}"] = s.index.to_numpy(dtype=float)
            d[f"p_{key}"] = (s / s.sum()).to_numpy()
            d[f"n_{key}"] = int(s.sum())
            d[f"mean_{key}"] = float((d[f"vals_{key}"] * d[f"p_{key}"]).sum())
        d["ref_gap"] = d["mean_f"] - d["mean_c"]
        out.append(d)
    return out


def simulate(dists: list[dict], ns: list[int] = N_GRID, tolerances: list[float] = TOLERANCES,
             n_draws: int = N_DRAWS, seed: int = SEED, verbose: bool = True) -> np.ndarray:
    """share[employer, n, tolerance] = fraction of draws whose gap is within +/-tolerance of the reference gap."""
    rng = np.random.default_rng(seed)
    share = np.empty((len(dists), len(ns), len(tolerances)))
    for i, d in enumerate(dists):
        for j, n in enumerate(ns):
            cf = rng.multinomial(n, d["p_f"], size=n_draws)
            cc = rng.multinomial(n, d["p_c"], size=n_draws)
            gap = (cf @ d["vals_f"] - cc @ d["vals_c"]) / n
            dev = np.abs(gap - d["ref_gap"])
            for k, tol in enumerate(tolerances):
                share[i, j, k] = (dev <= tol + 1e-9).mean()
        if verbose and (i + 1) % 50 == 0:
            print(f"  simulated {i + 1}/{len(dists)} reference employers")
    return share


def smallest_n(curve: np.ndarray, ns: list[int], level: float) -> float:
    """Smallest n whose share >= level (NaN if even the largest n does not reach it)."""
    ok = np.flatnonzero(curve >= level - 1e-12)
    return float(ns[ok[0]]) if len(ok) else np.nan


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def run_rq3(df: pd.DataFrame, n_draws: int = N_DRAWS, seed: int = SEED, verbose: bool = True) -> dict:
    sigma = float(df["rating"].std())
    n_formula = normal_approx_n(sigma)

    dists = reference_distributions(df)
    if verbose:
        print(f"Reference employers (>= {REF_MIN_EACH:,} former and current): {len(dists)}   [report: 333]")
    if not dists:
        raise SystemExit("No employer has enough reviews in both groups to be a reference employer.")
    share = simulate(dists, N_GRID, TOLERANCES, n_draws, seed, verbose)
    pooled = share.mean(axis=0)                                  # (n, tolerance)

    # curve table
    curve = pd.DataFrame({"n": N_GRID})
    for k, tol in enumerate(TOLERANCES):
        curve[f"share_within_{tol:.2f}"] = pooled[:, k]

    # employer-level counts for H3 (ALL employers, not only reference ones)
    cnt = df.groupby(["company", "former"], observed=True).size().unstack(fill_value=0)
    cnt = cnt.rename(columns={True: "n_former", False: "n_current"})
    cnt["n_min"] = cnt[["n_former", "n_current"]].min(axis=1)
    cnt["n_total"] = cnt["n_former"] + cnt["n_current"]

    def coverage(n_star: float) -> dict:
        if np.isnan(n_star):
            return {"share_employers_enough": np.nan, "share_reviews_covered": np.nan}
        enough = cnt["n_min"] >= n_star
        return {"share_employers_enough": float(enough.mean()),
                "share_reviews_covered": float(cnt.loc[enough, "n_total"].sum() / cnt["n_total"].sum())}

    # n* table: tolerance x level (main = 0.25 / 95%)
    rows = []
    for k, tol in enumerate(TOLERANCES):
        for level in LEVELS:
            n_star = smallest_n(pooled[:, k], N_GRID, level)
            rows.append({"tolerance": tol, "level": level, "n_star": n_star, **coverage(n_star),
                         "main": bool(tol == TOLERANCE and level == LEVEL)})
    nstar_table = pd.DataFrame(rows)
    main = nstar_table[nstar_table["main"]].iloc[0]
    n_star = main["n_star"]

    # per-employer n* (smallest n where THAT employer reaches the level)
    k_main = TOLERANCES.index(TOLERANCE)
    per_emp = pd.DataFrame({
        "firm_id": [d["firm_id"] for d in dists],
        "n_former": [d["n_f"] for d in dists], "n_current": [d["n_c"] for d in dists],
        "ref_gap": [d["ref_gap"] for d in dists],
        "n_star": [smallest_n(share[i, :, k_main], N_GRID, LEVEL) for i in range(len(dists))],
    })

    # reference-gap precision (the full-data gap is itself an estimate)
    margin_1000 = Z * sigma * np.sqrt(2 / REF_MIN_EACH)

    summary = {
        "n_reference_employers": len(dists), "sigma": sigma, "n_star_resampling": n_star,
        "n_star_normal_approx": n_formula,
        "ref_margin_at_1000": float(margin_1000),
        "n_employers_all": int(len(cnt)),
        **coverage(n_star),
        **{f"formula_{k_}": v for k_, v in coverage(float(n_formula)).items()},
    }
    # H3: rejected if at least half of employers have >= n* reviews in both groups
    summary["H3_supported"] = bool(summary["share_employers_enough"] < 0.5) if not np.isnan(n_star) else None
    summary["H3_supported_formula"] = bool(summary["formula_share_employers_enough"] < 0.5)
    summary["share_per_employer_nstar_above_formula"] = float(
        (per_emp["n_star"].fillna(np.inf) > n_formula).mean())
    return {"summary": summary, "curve": curve, "nstar_table": nstar_table, "employers": per_emp}


# --------------------------------------------------------------------------- #
# Reporting and figures
# --------------------------------------------------------------------------- #
def print_report(res: dict):
    s = res["summary"]
    fmt = lambda v: "not reached by n = 300" if pd.isna(v) else f"{int(v)}"
    print("\n" + "=" * 70 + "\nRQ3  How many reviews per group are needed to trust the gap?\n" + "=" * 70)
    print(f"Reference employers: {s['n_reference_employers']}   "
          f"(their own gap has a 95% margin of about +/-{s['ref_margin_at_1000']:.2f} at 1,000 reviews per group)")
    print(f"Rating SD (all reviews): {s['sigma']:.2f}")
    print(f"\nn* from resampling (tolerance {TOLERANCE}, {LEVEL:.0%}): {fmt(s['n_star_resampling'])}")
    print(f"n* from the normal approximation (baseline)  : {s['n_star_normal_approx']}   [report: 183]")
    print("\nSensitivity of n* (resampling):")
    t = res["nstar_table"].drop(columns="main").copy()
    t["n_star"] = t["n_star"].map(fmt)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nAmong ALL {s['n_employers_all']:,} employers, using the resampling n*:")
    print(f"  with >= n* reviews in both groups: {s['share_employers_enough']:.1%}   "
          f"(they hold {s['share_reviews_covered']:.1%} of reviews)")
    print(f"Using the formula n* = {s['n_star_normal_approx']}:")
    print(f"  with >= n* reviews in both groups: {s['formula_share_employers_enough']:.1%}   "
          f"(they hold {s['formula_share_reviews_covered']:.1%} of reviews)   [report: 6.7% and 80.1%]")
    h3 = s["H3_supported"]
    print(f"\nH3 (more than half of employers lack enough reviews): "
          f"{'SUPPORTED' if h3 else 'REJECTED' if h3 is not None else 'cannot be decided (n* not reached)'}"
          f"  (rejected if at least half of employers have >= n* in both groups)")
    print(f"Does resampling confirm the formula? resampling {fmt(s['n_star_resampling'])} vs formula "
          f"{s['n_star_normal_approx']}")
    print("Note: reference employers are the largest employers, so n* may not transfer to small ones.")


def plot_curve(res: dict, path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    c, s = res["curve"], res["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    for tol in TOLERANCES:
        y = c[f"share_within_{tol:.2f}"]
        if tol == TOLERANCE:
            ax.plot(c["n"], y, color=BLUE, lw=2.4)
        else:
            ax.plot(c["n"], y, color=INK_2, lw=1.2, ls="--")
        ax.text(c["n"].iloc[3], y.iloc[3], f"+/-{tol:.2f}", color=INK, fontsize=9, va="center", ha="center",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axhline(LEVEL, color=INK, lw=1, ls=":")
    handles = [Line2D([], [], color=BLUE, lw=2.4, label=f"Share of draws within +/-{TOLERANCE} (main)"),
               Line2D([], [], color=INK_2, lw=1.2, ls="--", label="Other tolerances"),
               Line2D([], [], color=INK, lw=1, ls=":", label=f"{LEVEL:.0%} target")]
    if not np.isnan(s["n_star_resampling"]):
        ax.axvline(s["n_star_resampling"], color=BLUE, lw=1.2)
        handles.append(Line2D([], [], color=BLUE, lw=1.2, label=f"n* resampling = {int(s['n_star_resampling'])}"))
    if s["n_star_normal_approx"] <= N_GRID[-1]:
        ax.axvline(s["n_star_normal_approx"], color=ORANGE, lw=1.6)
    handles.append(Line2D([], [], color=ORANGE, lw=1.6, label=f"n* normal approximation = {s['n_star_normal_approx']}"))
    ax.set_xlim(N_GRID[0], N_GRID[-1] + 5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Reviews per group (n)", color=INK_2)
    ax.set_ylabel("Share of draws close to the reference gap", color=INK_2)
    ax.set_title(f"RQ3: how many reviews are needed ({s['n_reference_employers']} reference employers)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def save_outputs(res: dict, outdir: Path, figdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res["summary"]]).to_csv(outdir / "rq3_summary.csv", index=False)
    res["curve"].to_csv(outdir / "rq3_curve.csv", index=False)
    res["nstar_table"].to_csv(outdir / "rq3_sensitivity.csv", index=False)
    res["employers"].to_csv(outdir / "rq3_reference_employers.csv", index=False)
    plot_curve(res, figdir / "rq3_curve.png")
    print(f"\nSaved tables to {outdir}/ and figures to {figdir}/")


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/processed.parquet", help="processed parquet file")
    ap.add_argument("--out", default="results", help="folder for CSV tables")
    ap.add_argument("--figures", default="figures", help="folder for PNG figures")
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="draws per (employer, n)")
    a = ap.parse_args()
    df, log = load_processed(a.data)
    print({k: v for k, v in log.items()})
    res = run_rq3(df, n_draws=a.draws)
    print_report(res)
    save_outputs(res, Path(a.out), Path(a.figures))


if __name__ == "__main__":
    main()
