"""
Visualizations for the joke tournament results.

Produces:
  - figures/winrate_ranking.png   — horizontal bar chart with 95% CI
  - figures/matchup_heatmap.png   — 9×9 win-rate heatmap
  - figures/position_bias.png     — position effect per category

Usage: python3.11 visualize.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from config import (
    CATEGORY_PALETTE,
    CATEGORY_SHORT_LABELS,
    FIGURES_DIR,
    POSITION_BIAS_RATES,
    TOURNAMENT_RESULTS_FILE,
)

FIGURES_DIR.mkdir(exist_ok=True)

with open(TOURNAMENT_RESULTS_FILE) as f:
    results = json.load(f)


# Aggregate global win rates
def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Calculate a Wilson score confidence interval for a proportion.

    Args:
        wins: Number of successful outcomes.
        total: Total number of trials.
        z: Z-score for the desired confidence level.

    Returns:
        Lower and upper confidence interval bounds.
    """
    if total == 0:
        return 0.0, 0.0
    p     = wins / total
    denom = 1 + z**2 / total
    c     = (p + z**2 / (2 * total)) / denom
    m     = z * np.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denom
    return c - m, c + m

cat_wins  = defaultdict(int)
cat_total = defaultdict(int)

for matchup, data in results.items():
    total = data["total_fights"]
    for cat, v in data.items():
        if cat == "total_fights":
            continue
        cat_wins[cat]  += v["wins"]
        cat_total[cat] += total

categories = sorted(cat_wins, key=lambda c: cat_wins[c]/cat_total[c], reverse=True)
rates  = [cat_wins[c] / cat_total[c] for c in categories]
errors = []
for c in categories:
    lo, hi = wilson_ci(cat_wins[c], cat_total[c])
    rate   = cat_wins[c] / cat_total[c]
    errors.append([rate - lo, hi - rate])
errors = np.array(errors).T   # shape (2, n)


# Fig 1 — Win-rate ranking bar chart
fig, ax = plt.subplots(figsize=(9, 5))

colors = [CATEGORY_PALETTE[c] for c in categories]
labels = [CATEGORY_SHORT_LABELS[c] for c in categories]
x      = np.arange(len(categories))

bars = ax.bar(x, rates, color=colors, width=0.6, zorder=3)
ax.errorbar(x, rates, yerr=errors, fmt="none", color="black",
            capsize=4, linewidth=1.2, zorder=4)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5,
           label="No bias (50%)")

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{rate:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Global win rate", fontsize=11)
ax.set_title("Which joke category does Claude find funniest?\n"
             "Global win rate across all matchups (95% CI, ~880 fights per matchup)",
             fontsize=11, pad=12)
ax.set_ylim(0.28, 0.78)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "winrate_ranking.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ figures/winrate_ranking.png")



# Fig 2 — 9×9 heatmap

# Order rows/cols by global win rate (best first)
ordered = categories   # already sorted above

# Build matrix: matrix[i][j] = win rate of ordered[i] against ordered[j]
n   = len(ordered)
idx = {c: i for i, c in enumerate(ordered)}
matrix = np.full((n, n), np.nan)

for matchup, data in results.items():
    cats = {k: v for k, v in data.items() if k != "total_fights"}
    total = data["total_fights"]
    for cat, v in cats.items():
        # find the opponent
        other = [k for k in cats if k != cat][0]
        i, j  = idx[cat], idx[other]
        matrix[i][j] = v["wins"] / total

fig, ax = plt.subplots(figsize=(10, 8))

mask = np.isnan(matrix)  # mask diagonal (NaN)
cmap = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(
    matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0.5,
    vmin=0.2, vmax=0.8,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"label": "Win rate (row beats column)", "shrink": 0.8},
)

short_labels = [CATEGORY_SHORT_LABELS[c] for c in ordered]
ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(short_labels, rotation=0, fontsize=9)
ax.set_title("Head-to-head win rates between joke categories\n"
             "(cell = probability that row category beats column category)",
             fontsize=11, pad=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "matchup_heatmap.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ figures/matchup_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Position bias per category
# ═══════════════════════════════════════════════════════════════════════════════

# Sort by global win rate to keep consistent ordering
pos_cats  = categories   # same order as figure 1
pos1_vals = [POSITION_BIAS_RATES[c][0] for c in pos_cats]
pos2_vals = [POSITION_BIAS_RATES[c][1] for c in pos_cats]
short_pos = [CATEGORY_SHORT_LABELS[c] for c in pos_cats]

x    = np.arange(len(pos_cats))
w    = 0.35
fig, ax = plt.subplots(figsize=(10, 5))

b1 = ax.bar(x - w/2, pos1_vals, w, label="Position 1 (first read)",
            color="#3498db", alpha=0.85, zorder=3)
b2 = ax.bar(x + w/2, pos2_vals, w, label="Position 2 (last read)",
            color="#e74c3c", alpha=0.85, zorder=3)

ax.axhline(0.5,  color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.axhline(0.35, color="#3498db", linestyle=":", linewidth=1, alpha=0.6,
           label="Pos1 baseline (35%)")
ax.axhline(0.65, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.6,
           label="Pos2 baseline (65%)")

ax.set_xticks(x)
ax.set_xticklabels(short_pos, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Win rate", fontsize=11)
ax.set_title("Position bias per category\n"
             "Win rate when the joke is presented first (pos1) vs last (pos2)",
             fontsize=11, pad=12)
ax.set_ylim(0.1, 0.88)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.legend(fontsize=9, loc="upper right")
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "position_bias.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ figures/position_bias.png")

print("\nAll figures saved to figures/")
