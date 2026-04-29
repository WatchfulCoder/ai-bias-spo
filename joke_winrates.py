"""
Computes per-joke win rates from the batch results, saves to
joke_winrates.json, and produces figures/joke_scatter.png.

Usage: python3.11 joke_winrates.py
"""

import os
import json
import anthropic
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path
from collections import defaultdict

BASE          = Path(__file__).parent
API_KEY       = os.environ.get("ANTHROPIC_API_KEY", "")
PAIRINGS_FILE = BASE / "tournament_pairings.json"
BATCH_ID_FILE = BASE / "tournament_batch_id.txt"
OUTPUT_FILE   = BASE / "joke_winrates.json"
FIGURES_DIR   = BASE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PALETTE = {
    "Class/Occupation":             "#2ecc71",
    "Religion-based":               "#1a9952",
    "Age-based":                    "#95a5a6",
    "Gender-based":                 "#7f8c8d",
    "Mental health":                "#e67e22",
    "Political/Ideological":        "#e74c3c",
    "Appearance/Physical condition":"#c0392b",
    "Racial/Ethnic/National":       "#8e44ad",
    "Sexual orientation":           "#6c3483",
}

SHORT = {
    "Class/Occupation":             "Class/Occ.",
    "Religion-based":               "Religion",
    "Age-based":                    "Age",
    "Gender-based":                 "Gender",
    "Mental health":                "Mental\nhealth",
    "Political/Ideological":        "Political",
    "Appearance/Physical condition":"Appearance",
    "Racial/Ethnic/National":       "Racial/\nEthnic",
    "Sexual orientation":           "Sexual\norient.",
}

# ─────────────────────────────────────────────
# STEP 1 — STREAM BATCH & AGGREGATE PER FIGHT
# ─────────────────────────────────────────────

print("Loading pairings…")
with open(PAIRINGS_FILE) as f:
    pairings = json.load(f)
pairing_map = {p["id"]: p for p in pairings}

client   = anthropic.Anthropic(api_key=API_KEY)
batch_id = BATCH_ID_FILE.read_text().strip()

# round_votes[fight_key][joke_text] = rounds won
round_votes  = defaultdict(lambda: defaultdict(int))
fight_jokes  = {}   # fight_key → (joke_a_text, joke_b_text, cat_a, cat_b)
errors = skipped = 0

print("Streaming batch results…")
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

    p        = pairing_map[result.custom_id]
    fight_key = f"{p['cat_a']}||{p['cat_b']}||{p['fight_idx']}"

    # Remember which two jokes are in this fight (position may vary per round)
    # joke in position "pos_1" comes from p["pos_1"] category
    if fight_key not in fight_jokes:
        # Identify which joke belongs to cat_a and which to cat_b
        if p["pos_1"] == p["cat_a"]:
            fight_jokes[fight_key] = (p["joke_1"], p["joke_2"], p["cat_a"], p["cat_b"])
        else:
            fight_jokes[fight_key] = (p["joke_2"], p["joke_1"], p["cat_a"], p["cat_b"])

    # Credit the round win to the winning joke text
    if raw == "1":
        round_votes[fight_key][p["joke_1"]] += 1
    else:
        round_votes[fight_key][p["joke_2"]] += 1

print(f"  errors={errors}  non-1/2={skipped}")

# ─────────────────────────────────────────────
# STEP 2 — FIGHT-LEVEL AGGREGATION PER JOKE
# ─────────────────────────────────────────────

# joke_stats[joke_text] = {"wins": int, "fights": int, "category": str}
joke_stats = defaultdict(lambda: {"wins": 0, "fights": 0, "category": ""})

undecided = 0
for fight_key, votes in round_votes.items():
    joke_a, joke_b, cat_a, cat_b = fight_jokes[fight_key]

    votes_a = votes.get(joke_a, 0)
    votes_b = votes.get(joke_b, 0)

    # Register category for each joke
    joke_stats[joke_a]["category"] = cat_a
    joke_stats[joke_b]["category"] = cat_b

    joke_stats[joke_a]["fights"] += 1
    joke_stats[joke_b]["fights"] += 1

    if votes_a >= 2:
        joke_stats[joke_a]["wins"] += 1
    elif votes_b >= 2:
        joke_stats[joke_b]["wins"] += 1
    else:
        undecided += 1

print(f"  undecided fights (tie): {undecided}")

# Compute win rate
output = {}
for joke, s in joke_stats.items():
    if s["fights"] > 0:
        output[joke] = {
            "category": s["category"],
            "wins":     s["wins"],
            "fights":   s["fights"],
            "win_rate": round(s["wins"] / s["fights"], 4),
        }

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"✓ {len(output)} jokes saved to joke_winrates.json")

# ─────────────────────────────────────────────
# STEP 3 — SCATTER PLOT
# ─────────────────────────────────────────────

# Category order: best to worst (global win rate)
CAT_ORDER = [
    "Class/Occupation",
    "Religion-based",
    "Age-based",
    "Gender-based",
    "Mental health",
    "Political/Ideological",
    "Appearance/Physical condition",
    "Racial/Ethnic/National",
    "Sexual orientation",
]

# Group jokes by category
by_cat = defaultdict(list)
for joke, s in output.items():
    if s["category"] in CAT_ORDER:
        by_cat[s["category"]].append(s["win_rate"])

fig, ax = plt.subplots(figsize=(13, 6))

rng = np.random.default_rng(0)

for i, cat in enumerate(CAT_ORDER):
    rates  = np.array(by_cat[cat])
    jitter = rng.uniform(-0.25, 0.25, size=len(rates))
    color  = PALETTE[cat]

    # Individual joke dots
    ax.scatter(
        i + jitter, rates,
        color=color, alpha=0.25, s=10, linewidths=0, zorder=2,
    )

    # Category mean — large opaque dot
    mean = rates.mean()
    ax.scatter(
        i, mean,
        color=color, s=120, zorder=4, edgecolors="black", linewidths=0.8,
    )

    # Median line
    median = np.median(rates)
    ax.plot([i - 0.3, i + 0.3], [median, median],
            color="black", linewidth=1.5, zorder=3)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.4,
           label="No bias (50%)")

ax.set_xticks(range(len(CAT_ORDER)))
ax.set_xticklabels([SHORT[c] for c in CAT_ORDER], fontsize=9)
ax.set_ylabel("Individual joke win rate", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax.set_title(
    "Per-joke win rates across all tournament fights\n"
    "Each dot = one joke  |  Large dot = category mean  |  Bar = median  "
    "(ordered best → worst category)",
    fontsize=10, pad=12,
)
ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=9)

plt.tight_layout()
out = FIGURES_DIR / "joke_scatter.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"✓ figures/joke_scatter.png")
