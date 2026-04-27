"""
Statistical analysis of tournament results.

Part 1 — Chi-squared tests + global win rates (uses tournament_results.json)
Part 2 — Position bias analysis (re-streams raw batch results from API)

Usage: python analysis.py
"""

import os
import json
import anthropic
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

BASE          = Path(__file__).parent
API_KEY       = os.environ.get("ANTHROPIC_API_KEY", "")
RESULTS_FILE  = BASE / "tournament_results.json"
PAIRINGS_FILE = BASE / "tournament_pairings.json"
BATCH_ID_FILE = BASE / "tournament_batch_id.txt"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def wilson_ci(wins, total, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return centre - margin, centre + margin

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─────────────────────────────────────────────
# PART 1 — GLOBAL WIN RATES + CHI-SQUARED
# ─────────────────────────────────────────────

def run_stats():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    # Aggregate wins and fights per category across all matchups
    cat_wins  = defaultdict(int)
    cat_total = defaultdict(int)

    for matchup, data in results.items():
        total = data["total_fights"]
        for cat, v in data.items():
            if cat == "total_fights":
                continue
            cat_wins[cat]  += v["wins"]
            cat_total[cat] += total   # each fight counts once per side

    # ── Global ranking ──
    print("=" * 65)
    print("GLOBAL WIN RATES (binomial test vs H0: win_rate = 0.50)")
    print("=" * 65)
    ranking = sorted(cat_wins, key=lambda c: cat_wins[c] / cat_total[c], reverse=True)
    for rank, cat in enumerate(ranking, 1):
        w = cat_wins[cat]
        t = cat_total[cat]
        rate = w / t
        lo, hi = wilson_ci(w, t)
        p = stats.binomtest(w, t, p=0.5).pvalue
        print(f"  {rank}. {cat:<35} {rate:.3f}  [{lo:.3f}–{hi:.3f}]  p={p:.2e} {sig_stars(p)}")

    # ── Per-matchup chi-squared ──
    print()
    print("=" * 65)
    print("CHI-SQUARED PER MATCHUP  (H0: 50/50 split)")
    print("=" * 65)
    for matchup, data in sorted(results.items()):
        total = data["total_fights"]
        cats  = {k: v for k, v in data.items() if k != "total_fights"}
        names = list(cats.keys())          # already sorted by wins (desc)
        observed = [cats[c]["wins"] for c in names]
        expected = [total / 2, total / 2]
        chi2, p  = stats.chisquare(observed, f_exp=expected)
        rate     = observed[0] / total
        print(
            f"  {names[0]:<30} > {names[1]:<30}"
            f"  {rate:.3f}  chi2={chi2:6.1f}  p={p:.2e} {sig_stars(p)}"
        )


# ─────────────────────────────────────────────
# PART 2 — POSITION BIAS
# ─────────────────────────────────────────────

def run_position_bias():
    if not BATCH_ID_FILE.exists():
        print("No batch ID found — skipping position bias.")
        return

    with open(PAIRINGS_FILE) as f:
        pairings = json.load(f)
    pairing_map = {p["id"]: p for p in pairings}

    client   = anthropic.Anthropic(api_key=API_KEY)
    batch_id = BATCH_ID_FILE.read_text().strip()

    # Counters
    chose_1 = chose_2 = errors = skipped = 0

    # Per-category: wins & appearances in each position
    cat_pos = defaultdict(lambda: {
        "pos1_wins": 0, "pos1_total": 0,
        "pos2_wins": 0, "pos2_total": 0,
    })

    print("\nStreaming batch results for position bias…")
    for result in client.beta.messages.batches.results(
        batch_id, betas=["message-batches-2024-09-24"]
    ):
        if result.result.type != "succeeded":
            errors += 1
            continue

        raw = result.result.message.content[0].text.strip().rstrip(".").strip()
        if raw not in ("1", "2"):
            skipped += 1
            continue

        p    = pairing_map[result.custom_id]
        cat1 = p["pos_1"]   # category whose joke sits in slot 1
        cat2 = p["pos_2"]   # category whose joke sits in slot 2

        cat_pos[cat1]["pos1_total"] += 1
        cat_pos[cat2]["pos2_total"] += 1

        if raw == "1":
            chose_1 += 1
            cat_pos[cat1]["pos1_wins"] += 1
        else:
            chose_2 += 1
            cat_pos[cat2]["pos2_wins"] += 1

    total_rounds = chose_1 + chose_2

    # ── Overall position bias ──
    print()
    print("=" * 65)
    print("OVERALL POSITION BIAS")
    print("=" * 65)
    p_binom = stats.binomtest(chose_1, total_rounds, p=0.5).pvalue
    print(f"  Chose position 1 : {chose_1}/{total_rounds} = {chose_1/total_rounds:.4f}")
    print(f"  Chose position 2 : {chose_2}/{total_rounds} = {chose_2/total_rounds:.4f}")
    print(f"  Binomial test    : p = {p_binom:.4f}  {sig_stars(p_binom)}")
    print(f"  (errors={errors}, non-1/2 responses={skipped})")

    # ── Per-category position effect ──
    print()
    print("=" * 65)
    print("POSITION EFFECT PER CATEGORY")
    print("(win rate when joke is in slot 1  vs  slot 2)")
    print("=" * 65)
    for cat in sorted(cat_pos):
        d  = cat_pos[cat]
        r1 = d["pos1_wins"] / d["pos1_total"] if d["pos1_total"] else 0
        r2 = d["pos2_wins"] / d["pos2_total"] if d["pos2_total"] else 0
        diff = r1 - r2
        # Chi-squared 2×2: [[pos1_wins, pos1_losses], [pos2_wins, pos2_losses]]
        pos1_loss = d["pos1_total"] - d["pos1_wins"]
        pos2_loss = d["pos2_total"] - d["pos2_wins"]
        contingency = [[d["pos1_wins"], pos1_loss],
                       [d["pos2_wins"], pos2_loss]]
        chi2, p, *_ = stats.chi2_contingency(contingency)
        print(
            f"  {cat:<35}  pos1={r1:.3f}  pos2={r2:.3f}"
            f"  Δ={diff:+.3f}  p={p:.2e} {sig_stars(p)}"
        )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_stats()
    run_position_bias()
