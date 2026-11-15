#!/usr/bin/env python3
"""
RQ2 - At the same rating, does Cons text still distinguish former from current
      employees, and what words drive it?

Implements Sections 3.1 and 4.2 of the midterm report:

  * split ALL employers into train / validation / test = 64 / 16 / 20 (no employer
    appears in more than one set, so the model cannot learn employer-specific words)
  * remove phrases that state the status outright ("I left", "laid off", ...)
  * TF-IDF (scikit-learn definition) + logistic regression (cross-entropy loss);
    vocabulary and model are fitted on the TRAIN set only
  * regularization strength C is tuned on the VALIDATION set; the TEST set is used once
  * evaluation = AUC on pairs of one former and one current review from the same
    test employer with the same rating (share of pairs where the former review
    gets the higher score; 0.50 = chance), averaged across employers
  * 95% CI = bootstrap over test employers
  * baseline = logistic regression on the number of words in Cons, same pairs
  * H2a is rejected if the 95% CI of the AUC includes 0.5
  * H2b is rejected if "management" is not among the 20 words with the largest
    coefficients for "former"
  * also checks whether past-tense words ("was", "used", ...) show up in the top words

Usage (run from the repo root; the three rq*.py files must sit in the same folder)
-----
    python rq2_text_signal.py --data data/processed/processed.parquet
    python rq2_text_signal.py --data data/processed/processed.parquet --max-train 1000000   # faster

In a notebook:
    import sys; sys.path.append("..")
    from rq2_text_signal import run_rq2, print_report
    from rq1_rating_gap import load_processed
    df, log = load_processed("../data/processed/processed.parquet", extra=("cons",))
    res = run_rq2(df); print_report(res)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rq1_rating_gap import (BAR, BLUE, GRID, INK, INK_2, ORANGE, SEED, SURFACE, _style,
                            bootstrap_mean_ci, load_processed)

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
TEST_SHARE = 0.20            # of all employers
VAL_SHARE = 0.20             # of the remaining employers  -> 64 / 16 / 20 overall
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]      # regularization strengths tried on validation
MIN_DF = 20                  # a word must appear in at least 20 training reviews
MAX_FEATURES = 50_000
TOP_N = 20
TARGET_WORD = "management"   # H2b
CHANCE = 0.5

# Phrases that state the status outright. Removed from Cons before fitting (Sec. 4.2).
# Edit this list if you find more giveaways among the top words.
SHORTCUT_PATTERNS = [
    r"\bi\s+left\b", r"\blaid\s*off\b", r"\bi\s+quit\b", r"\bi\s+resigned\b",
    r"\bwas\s+fired\b", r"\bi\s+was\s+let\s+go\b", r"\bno\s+longer\s+work(?:s|ed)?\b",
    r"\bformer\s+employee\b", r"\bex[- ]employee\b", r"\bwhen\s+i\s+worked\b",
    r"\bused\s+to\s+work\b",
]
# Past-tense wording that may leak status; only CHECKED among the top words (Sec. 4.2).
PAST_TENSE_WORDS = {"was", "were", "had", "did", "used", "left", "got", "been", "would", "didn", "wasn"}


# --------------------------------------------------------------------------- #
# Splitting and text preparation
# --------------------------------------------------------------------------- #
def split_employers(df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Return an int8 array (0 train, 1 validation, 2 test), one value per review, by employer."""
    codes = df["company"].cat.codes.to_numpy()
    n_firms = len(df["company"].cat.categories)
    perm = np.random.default_rng(seed).permutation(n_firms)
    n_test = int(round(TEST_SHARE * n_firms))
    rest = perm[n_test:]
    n_val = int(round(VAL_SHARE * len(rest)))
    firm_split = np.zeros(n_firms, dtype=np.int8)
    firm_split[perm[:n_test]] = 2
    firm_split[rest[:n_val]] = 1
    return firm_split[codes]


_SHORTCUT = re.compile("|".join(SHORTCUT_PATTERNS), flags=re.IGNORECASE)


def remove_shortcuts(cons: pd.Series) -> tuple[pd.Series, int]:
    """Delete status-revealing phrases; returns the cleaned text and how many reviews changed."""
    text = cons.fillna("").astype(str)
    cleaned = text.str.replace(_SHORTCUT, " ", regex=True)
    return cleaned, int((cleaned != text).sum())


def pairable(df: pd.DataFrame) -> pd.Series:
    """Rows whose (employer, rating) cell contains BOTH a former and a current review."""
    m = df.groupby(["company", "rating"], observed=True)["former"].transform("mean")
    return (m > 0) & (m < 1)


# --------------------------------------------------------------------------- #
# Evaluation: same-employer, same-rating pair AUC
# --------------------------------------------------------------------------- #
def employer_pair_auc(codes: np.ndarray, rating: np.ndarray, former: np.ndarray,
                      score: np.ndarray) -> pd.DataFrame:
    """
    For every employer: share of (former, current) pairs with the SAME rating where the
    former review gets the higher score (ties count 1/2). Computed exactly with ranks.
    Returns a DataFrame indexed by employer code with columns auc and n_pairs.
    """
    key = codes.astype(np.int64) * 8 + rating.astype(np.int64)      # ratings are 1..5
    order = np.argsort(key, kind="stable")
    key_s, f_s, sc_s = key[order], former[order].astype(bool), score[order]
    cuts = np.flatnonzero(np.diff(key_s)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(key_s)]
    wins: dict[int, float] = {}
    pairs: dict[int, float] = {}
    for a, b in zip(starts, ends):
        f = f_s[a:b]
        nf = int(f.sum())
        nc = (b - a) - nf
        if nf == 0 or nc == 0:
            continue
        ranks = rankdata(sc_s[a:b])                                  # average ranks handle ties
        u = ranks[f].sum() - nf * (nf + 1) / 2                       # Mann-Whitney U = wins + 0.5 ties
        firm = int(key_s[a] // 8)
        wins[firm] = wins.get(firm, 0.0) + u
        pairs[firm] = pairs.get(firm, 0.0) + nf * nc
    out = pd.DataFrame({"n_pairs": pd.Series(pairs)})
    out["auc"] = pd.Series(wins) / out["n_pairs"]
    out.index.name = "firm_code"
    return out


def _mean_pair_auc(part: pd.DataFrame, score: np.ndarray) -> float:
    r = employer_pair_auc(part["company"].cat.codes.to_numpy(), part["rating"].to_numpy(),
                          part["former"].to_numpy(), score)
    return float(r["auc"].mean())


# --------------------------------------------------------------------------- #
# Main analysis
# --------------------------------------------------------------------------- #
def run_rq2(df: pd.DataFrame, max_train: int | None = None, seed: int = SEED,
            verbose: bool = True) -> dict:
    """df needs: company (category), rating, former (bool), cons (text)."""
    say = print if verbose else (lambda *a, **k: None)
    df = df.reset_index(drop=True)
    df["n_words"] = df["cons"].fillna("").astype(str).str.split().str.len().astype("float32")
    split = split_employers(df, seed)
    info = {"train": int((split == 0).sum()), "validation": int((split == 1).sum()),
            "test": int((split == 2).sum())}
    say(f"Split by employer -> reviews: train {info['train']:,}, validation {info['validation']:,}, "
        f"test {info['test']:,}")

    train = df[split == 0]
    if max_train and len(train) > max_train:
        train = train.sample(max_train, random_state=seed)
        say(f"Training on a random sample of {max_train:,} training reviews (--max-train)")
    val = df[split == 1]
    val = val[pairable(val).to_numpy()]
    test = df[split == 2]
    test = test[pairable(test).to_numpy()]

    # ---- text: remove status giveaways, fit TF-IDF on TRAIN only ------------------------------
    tr_text, n_changed = remove_shortcuts(train["cons"])
    say(f"Removed status-revealing phrases from {n_changed:,} training reviews")
    vec = TfidfVectorizer(min_df=MIN_DF, max_features=MAX_FEATURES, dtype=np.float32)
    X_tr = vec.fit_transform(tr_text)
    y_tr = train["former"].to_numpy().astype(int)
    X_va = vec.transform(remove_shortcuts(val["cons"])[0])
    X_te = vec.transform(remove_shortcuts(test["cons"])[0])
    say(f"Vocabulary: {X_tr.shape[1]:,} words")

    # ---- tune C on validation (mean same-employer, same-rating pair AUC) ----------------------
    rows, best = [], None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=300).fit(X_tr, y_tr)
        auc = _mean_pair_auc(val, model.decision_function(X_va))
        rows.append({"C": c, "validation_auc": auc})
        say(f"  C={c:<5} validation AUC = {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, c, model)
    _, best_c, model = best
    say(f"Chosen C = {best_c}")

    # ---- baseline: logistic regression on word count only ------------------------------------
    scaler = StandardScaler().fit(train[["n_words"]])
    base = LogisticRegression().fit(scaler.transform(train[["n_words"]]), y_tr)

    # ---- single evaluation on the TEST employers ---------------------------------------------
    codes, rating, former = (test["company"].cat.codes.to_numpy(), test["rating"].to_numpy(),
                             test["former"].to_numpy())
    tfidf_auc = employer_pair_auc(codes, rating, former, model.decision_function(X_te))["auc"]
    base_auc = employer_pair_auc(codes, rating, former,
                                 base.decision_function(scaler.transform(test[["n_words"]])))["auc"]
    both = pd.concat({"tfidf": tfidf_auc, "wordcount": base_auc}, axis=1).dropna()
    both["diff"] = both["tfidf"] - both["wordcount"]
    cats = test["company"].cat.categories
    both.index = cats[both.index.to_numpy()]
    both.index.name = "firm_id"

    summary = {"n_test_employers": len(both), "chosen_C": best_c}
    for name, col in (("tfidf", "tfidf"), ("wordcount", "wordcount"), ("diff", "diff")):
        lo, hi = bootstrap_mean_ci(both[col].to_numpy())
        summary[f"{name}_mean"], summary[f"{name}_ci_low"], summary[f"{name}_ci_high"] = \
            float(both[col].mean()), lo, hi
    summary["H2a_supported"] = bool(summary["tfidf_ci_low"] > CHANCE)       # rejected if CI includes 0.5
    summary["tfidf_beats_wordcount"] = bool(summary["diff_ci_low"] > 0)

    # ---- top words ---------------------------------------------------------------------------
    words = vec.get_feature_names_out()
    coef = model.coef_[0]
    order = np.argsort(-coef)
    pos, neg = order[coef[order] > 0][:TOP_N], order[::-1][coef[order[::-1]] < 0][:TOP_N]
    top_former = pd.DataFrame({"rank": np.arange(1, len(pos) + 1), "word": words[pos],
                               "coef": coef[pos], "points_to": "former"})
    top_current = pd.DataFrame({"rank": np.arange(1, len(neg) + 1), "word": words[neg],
                                "coef": coef[neg], "points_to": "current"})
    rank_of = {w: i + 1 for i, w in enumerate(words[order])}
    summary["target_word_rank"] = rank_of.get(TARGET_WORD)          # None if not in vocabulary
    summary["H2b_supported"] = bool(summary["target_word_rank"] and summary["target_word_rank"] <= TOP_N)
    summary["past_tense_in_top_former"] = sorted(set(top_former["word"]) & PAST_TENSE_WORDS)

    return {"summary": summary, "employer_auc": both, "validation": pd.DataFrame(rows),
            "top_former": top_former, "top_current": top_current, "split_info": info}


# --------------------------------------------------------------------------- #
# Reporting and figures
# --------------------------------------------------------------------------- #
def print_report(res: dict):
    s = res["summary"]
    print("\n" + "=" * 70 + "\nRQ2  Does Cons text separate former from current (same employer, same rating)?\n" + "=" * 70)
    print(f"Test employers with at least one pair: {s['n_test_employers']:,}   (chosen C = {s['chosen_C']})")
    print(f"TF-IDF logistic regression AUC : {s['tfidf_mean']:.3f}  95% CI [{s['tfidf_ci_low']:.3f}, {s['tfidf_ci_high']:.3f}]")
    print(f"Word-count baseline AUC        : {s['wordcount_mean']:.3f}  95% CI [{s['wordcount_ci_low']:.3f}, {s['wordcount_ci_high']:.3f}]")
    print(f"Difference (TF-IDF - baseline) : {s['diff_mean']:+.3f}  95% CI [{s['diff_ci_low']:+.3f}, {s['diff_ci_high']:+.3f}]")
    print(f"\nH2a (text separates former from current): {'SUPPORTED' if s['H2a_supported'] else 'REJECTED'}  "
          f"(rejected if the AUC CI includes {CHANCE})")
    print(f"TF-IDF adds signal beyond word count: {s['tfidf_beats_wordcount']}")
    r = s["target_word_rank"]
    print(f"H2b ('{TARGET_WORD}' in top {TOP_N}): {'SUPPORTED' if s['H2b_supported'] else 'REJECTED'}  "
          f"(rank = {r if r else 'not in vocabulary'})")
    print(f"Past-tense words among the top {TOP_N} 'former' words: {s['past_tense_in_top_former'] or 'none'}")
    print(f"\nTop {TOP_N} words pointing to FORMER:  " + ", ".join(res["top_former"]["word"]))
    print(f"Top {TOP_N} words pointing to CURRENT: " + ", ".join(res["top_current"]["word"]))


def plot_auc(res: dict, path: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    s = res["summary"]
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    _style(ax)
    for x, key, color in ((0, "tfidf", BLUE), (1, "wordcount", ORANGE)):
        m, lo, hi = s[f"{key}_mean"], s[f"{key}_ci_low"], s[f"{key}_ci_high"]
        ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color, capsize=4, markersize=8,
                    elinewidth=1.8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(x + 0.08, m, f"{m:.3f}", color=INK, fontsize=9, va="center")
    ax.axhline(CHANCE, color=INK, lw=1, ls="--")
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1], ["TF-IDF\nlogistic regression", "Word count\n(baseline)"])
    ax.set_ylabel("AUC (same employer, same rating pairs)", color=INK_2)
    ax.set_title(f"RQ2: AUC on {s['n_test_employers']:,} test employers", color=INK, fontsize=11, loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1, ls="--", label="0.50 = chance")],
              loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_top_words(res: dict, path: Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2), facecolor=SURFACE)
    for ax, key, color, title in ((axes[0], "top_former", BLUE, "Words pointing to FORMER"),
                                  (axes[1], "top_current", ORANGE, "Words pointing to CURRENT")):
        t = res[key].iloc[::-1]
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.barh(t["word"], t["coef"].abs(), color=color, height=0.7)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("|logistic regression coefficient|", color=INK_2)
        for lab in ax.get_yticklabels():
            lab.set_color(INK)
            if lab.get_text() == TARGET_WORD:
                lab.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def save_outputs(res: dict, outdir: Path, figdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res["summary"]]).to_csv(outdir / "rq2_summary.csv", index=False)
    res["employer_auc"].to_csv(outdir / "rq2_employer_auc.csv")
    res["validation"].to_csv(outdir / "rq2_validation.csv", index=False)
    pd.concat([res["top_former"], res["top_current"]]).to_csv(outdir / "rq2_top_words.csv", index=False)
    plot_auc(res, figdir / "rq2_auc.png")
    plot_top_words(res, figdir / "rq2_top_words.png")
    print(f"\nSaved tables to {outdir}/ and figures to {figdir}/")


def main():
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/processed.parquet", help="processed parquet file")
    ap.add_argument("--out", default="results", help="folder for CSV tables")
    ap.add_argument("--figures", default="figures", help="folder for PNG figures")
    ap.add_argument("--max-train", type=int, default=0,
                    help="optional cap on training reviews, for a faster run (0 = use all)")
    a = ap.parse_args()
    df, log = load_processed(a.data, extra=("cons",))
    print({k: v for k, v in log.items()})
    res = run_rq2(df, max_train=a.max_train or None)
    print_report(res)
    save_outputs(res, Path(a.out), Path(a.figures))


if __name__ == "__main__":
    main()
