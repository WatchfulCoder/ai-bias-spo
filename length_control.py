"""
Length control analysis.

Checks whether winning categories simply have longer / more complex jokes,
which would partially explain the tournament results independently of bias.

Metrics per category:
  - mean & median character length
  - mean & median word count
  - Pearson correlation between category mean length and tournament win rate
  - Kruskal-Wallis test (are length distributions significantly different?)
  - Produces figures/length_control.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE = Path(__file__).parent

# ── Tournament win rates (from previous analysis) ────────────────────────────
WIN_RATES = {
    "Class/Occupation":             0.662,
    "Religion-based":               0.627,
    "Age-based":                    0.531,
    "Gender-based":                 0.504,
    "Mental health":                0.475,
    "Political/Ideological":        0.466,
    "Appearance/Physical condition":0.454,
    "Racial/Ethnic/National":       0.383,
    "Sexual orientation":           0.378,
}

# ── Load annotated jokes ─────────────────────────────────────────────────────
df = pd.read_csv(BASE / "annotated_10k.csv", keep_default_na=False)

# Keep only single-label jokes in qualifying categories
is_single = (
    ~df["category_llm"].str.contains(";") &
    (df["category_llm"] != "None") &
    df["category_llm"].isin(WIN_RATES)
)
df = df[is_single].copy()

df["char_len"] = df["joke"].str.len()
df["word_len"] = df["joke"].str.split().str.len()

# ── Per-category stats ───────────────────────────────────────────────────────
rows = []
for cat, wr in WIN_RATES.items():
    sub = df[df["category_llm"] == cat]
    rows.append({
        "category":   cat,
        "win_rate":   wr,
        "n":          len(sub),
        "mean_chars": sub["char_len"].mean(),
        "med_chars":  sub["char_len"].median(),
        "mean_words": sub["word_len"].mean(),
        "med_words":  sub["word_len"].median(),
    })

stats_df = pd.DataFrame(rows).sort_values("win_rate", ascending=False)

print("=" * 70)
print("JOKE LENGTH BY CATEGORY (sorted by win rate)")
print("=" * 70)
print(f"{'Category':<35} {'WinRate':>8} {'MeanChars':>10} {'MeanWords':>10}")
print("-" * 70)
for _, r in stats_df.iterrows():
    print(f"  {r['category']:<33} {r['win_rate']:>8.3f} {r['mean_chars']:>10.0f} {r['mean_words']:>10.1f}")

# ── Correlation: win rate vs mean length ─────────────────────────────────────
r_chars, p_chars = stats.pearsonr(stats_df["win_rate"], stats_df["mean_chars"])
r_words, p_words = stats.pearsonr(stats_df["win_rate"], stats_df["mean_words"])

print()
print("=" * 70)
print("CORRELATION: win rate vs joke length")
print("=" * 70)
print(f"  Pearson r (win rate vs mean chars) : r={r_chars:+.3f}  p={p_chars:.3f}")
print(f"  Pearson r (win rate vs mean words) : r={r_words:+.3f}  p={p_words:.3f}")

if abs(r_chars) < 0.3 and p_chars > 0.05:
    print("\n  → No significant correlation. Length does NOT explain the bias.")
elif p_chars < 0.05:
    print(f"\n  → Significant correlation (p={p_chars:.3f}). Length partially confounds results.")
else:
    print("\n  → Weak correlation, not significant.")

# ── Kruskal-Wallis: are length distributions different across categories? ────
groups_chars = [df[df["category_llm"] == cat]["char_len"].values
                for cat in WIN_RATES]
kw_stat, kw_p = stats.kruskal(*groups_chars)

print()
print("=" * 70)
print("KRUSKAL-WALLIS: do categories differ in length distribution?")
print("=" * 70)
print(f"  H={kw_stat:.1f}  p={kw_p:.4f}")
if kw_p < 0.05:
    print("  → Categories have significantly different length distributions.")
    print("     (but check correlation above to see if this tracks win rates)")
else:
    print("  → No significant difference in length across categories.")

# ── Figure ───────────────────────────────────────────────────────────────────
SHORT = {
    "Class/Occupation":             "Class/Occ.",
    "Religion-based":               "Religion",
    "Age-based":                    "Age",
    "Gender-based":                 "Gender",
    "Mental health":                "Mental health",
    "Political/Ideological":        "Political",
    "Appearance/Physical condition":"Appearance",
    "Racial/Ethnic/National":       "Racial/Ethnic",
    "Sexual orientation":           "Sexual orient.",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: scatter win rate vs mean chars
ax = axes[0]
ax.scatter(stats_df["mean_chars"], stats_df["win_rate"],
           s=90, color="#2c7bb6", zorder=3)
for _, r in stats_df.iterrows():
    ax.annotate(SHORT[r["category"]],
                (r["mean_chars"], r["win_rate"]),
                textcoords="offset points", xytext=(6, 3), fontsize=8)
# Regression line
m, b = np.polyfit(stats_df["mean_chars"], stats_df["win_rate"], 1)
x_line = np.linspace(stats_df["mean_chars"].min(), stats_df["mean_chars"].max(), 100)
ax.plot(x_line, m * x_line + b, color="#e74c3c", linewidth=1.5, linestyle="--")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
ax.set_xlabel("Mean joke length (characters)", fontsize=10)
ax.set_ylabel("Tournament win rate", fontsize=10)
ax.set_title(f"Win rate vs. joke length (chars)\nr={r_chars:+.3f}, p={p_chars:.2f}",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# Right: boxplot of char lengths per category (sorted by win rate)
ax = axes[1]
ordered_cats = stats_df["category"].tolist()
data_for_box = [df[df["category_llm"] == cat]["char_len"].values
                for cat in ordered_cats]
bp = ax.boxplot(data_for_box, patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5))

colors_box = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(ordered_cats)))
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax.set_xticks(range(1, len(ordered_cats) + 1))
ax.set_xticklabels([SHORT[c] for c in ordered_cats],
                   rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Joke length (characters)", fontsize=10)
ax.set_title("Length distribution per category\n(ordered by win rate, best→worst)",
             fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Length control: does joke length explain tournament results?",
             fontsize=11, fontweight="bold", y=1.01)
plt.tight_layout()

out = BASE / "figures" / "length_control.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"\n✓ figures/length_control.png")
