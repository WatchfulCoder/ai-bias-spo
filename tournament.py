"""
Joke Tournament using Anthropic Batch API.

Usage:
    python tournament.py submit    -- pre-compute pairings & submit batch
    python tournament.py status    -- check batch progress
    python tournament.py retrieve  -- download results & compute scores
"""

import sys
import json
import random
import itertools
import anthropic
import pandas as pd
from collections import defaultdict
from typing import Any

from config import (
    ANNOTATED_JOKES_FILE,
    API_KEY,
    CATEGORY_ABBREVIATIONS,
    K_OPPONENTS,
    MIN_CATEGORY_SIZE,
    MODEL,
    N_ROUNDS,
    RANDOM_SEED,
    TOURNAMENT_BATCH_ID_FILE,
    TOURNAMENT_PAIRINGS_FILE,
    TOURNAMENT_PROMPT_TEMPLATE,
    TOURNAMENT_RESULTS_FILE,
)

CategoryJokes = dict[str, list[str]]
Pairing = dict[str, str | int]



def load_and_harmonize() -> tuple[CategoryJokes, int]:
    """Load and filter jokes into tournament-ready categories.

    Returns:
        A tuple containing the category-to-jokes mapping and the target sample
        size to use per category when building pairings.
    """
    df = pd.read_csv(ANNOTATED_JOKES_FILE, keep_default_na=False)

    # Keep only single-label jokes (no ";" = no multi-label, no None)
    is_single = ~df["category_llm"].str.contains(";") & (df["category_llm"] != "None")
    single = df[is_single]

    categories = {}
    print("Category sizes (single-label only):")
    for cat, group in single.groupby("category_llm"):
        n = len(group)
        if n >= MIN_CATEGORY_SIZE:
            categories[cat] = group["joke"].tolist()
            print(f"  ✓ {cat}: {n} jokes")
        else:
            print(f"  ✗ {cat}: {n} jokes — excluded (< {MIN_CATEGORY_SIZE})")

    MAX_TARGET = 154   # keeps total API calls under 100 000 (single batch limit)
    target = min(min(len(v) for v in categories.values()), MAX_TARGET)
    print(f"\nTarget size (capped at {MAX_TARGET}): {target} jokes")
    return categories, target


def build_pairings(categories: CategoryJokes, target: int) -> list[Pairing]:
    """Build round-level joke pairings for every ordered category matchup.

    Args:
        categories: Mapping of category names to the jokes available in each
            category.
        target: Number of jokes to sample from each category.

    Returns:
        A flat list of pairing dictionaries, one per tournament round request.
    """
    random.seed(RANDOM_SEED)
    pairings = []

    # permutations gives all ordered pairs: (A,B) AND (B,A)
    for cat_a, cat_b in itertools.permutations(categories.keys(), 2):
        jokes_a = random.sample(categories[cat_a], target)
        jokes_b = random.sample(categories[cat_b], target)

        fight_idx = 0
        for a_idx, joke_a in enumerate(jokes_a):
            # Pick K distinct opponents from B, cycling if needed
            start = (a_idx * K_OPPONENTS) % target
            opponents = jokes_b[start: start + K_OPPONENTS]
            if len(opponents) < K_OPPONENTS:
                opponents += jokes_b[: K_OPPONENTS - len(opponents)]

            for joke_b in opponents:
                for round_idx in range(N_ROUNDS):
                    if random.random() < 0.5:
                        joke_1, joke_2 = joke_a, joke_b
                        pos_1, pos_2   = cat_a, cat_b
                    else:
                        joke_1, joke_2 = joke_b, joke_a
                        pos_1, pos_2   = cat_b, cat_a

                    abbr_a = CATEGORY_ABBREVIATIONS.get(cat_a, cat_a[:4])
                    abbr_b = CATEGORY_ABBREVIATIONS.get(cat_b, cat_b[:4])
                    request_id = f"{abbr_a}_VS_{abbr_b}__f{fight_idx}__r{round_idx}"

                    pairings.append({
                        "id":        request_id,
                        "cat_a":     cat_a,
                        "cat_b":     cat_b,
                        "fight_idx": fight_idx,
                        "round_idx": round_idx,
                        "pos_1":     pos_1,
                        "pos_2":     pos_2,
                        "joke_1":    joke_1,
                        "joke_2":    joke_2,
                    })
                fight_idx += 1

    return pairings


def submit() -> None:
    """Submit a tournament batch to Anthropic.

    Returns:
        None.
    """
    print("=== LOADING & HARMONIZING ===")
    categories, target = load_and_harmonize()
    n_cats = len(categories)
    n_matchups = n_cats * (n_cats - 1)
    print(f"\n{n_cats} categories qualify → {n_matchups} matchups")

    print("\n=== BUILDING PAIRINGS ===")
    pairings = build_pairings(categories, target)
    n_fights = len(pairings) // N_ROUNDS
    print(f"{n_fights} fights × {N_ROUNDS} rounds = {len(pairings)} API calls")

    with open(TOURNAMENT_PAIRINGS_FILE, "w") as f:
        json.dump(pairings, f, ensure_ascii=False)
    print(f"Pairings saved to {TOURNAMENT_PAIRINGS_FILE}")

    print("\n=== SUBMITTING BATCH ===")
    requests: list[dict[str, Any]] = []
    for p in pairings:
        prompt = TOURNAMENT_PROMPT_TEMPLATE.format(joke_1=p["joke_1"], joke_2=p["joke_2"])
        requests.append({
            "custom_id": p["id"],
            "params": {
                "model":     MODEL,
                "max_tokens": 10,
                "messages":  [{"role": "user", "content": prompt}],
            },
        })

    client = anthropic.Anthropic(api_key=API_KEY)
    batch = client.beta.messages.batches.create(
        requests=requests,
        betas=["message-batches-2024-09-24"],
    )
    TOURNAMENT_BATCH_ID_FILE.write_text(batch.id)
    print(f"Batch submitted — ID: {batch.id} ({len(requests)} requests)")
    print(f"Batch ID saved to {TOURNAMENT_BATCH_ID_FILE}")


def status() -> None:
    """Print the current Anthropic tournament batch status.

    Returns:
        None.
    """
    if not TOURNAMENT_BATCH_ID_FILE.exists():
        print("No batch ID found. Run 'submit' first.")
        return
    client = anthropic.Anthropic(api_key=API_KEY)
    batch_id = TOURNAMENT_BATCH_ID_FILE.read_text().strip()
    batch = client.beta.messages.batches.retrieve(
        batch_id, betas=["message-batches-2024-09-24"]
    )
    print(f"Batch ID : {batch.id}")
    print(f"Status   : {batch.processing_status}")
    print(f"Counts   : {batch.request_counts}")
    if batch.processing_status == "ended":
        print("→ Ready. Run 'retrieve' to get results.")


def retrieve() -> None:
    """Retrieve tournament results and compute category win rates.

    Returns:
        None.
    """
    if not TOURNAMENT_BATCH_ID_FILE.exists():
        print("No batch ID found. Run 'submit' first.")
        return

    with open(TOURNAMENT_PAIRINGS_FILE) as f:
        pairings = json.load(f)
    pairing_map = {p["id"]: p for p in pairings}

    client = anthropic.Anthropic(api_key=API_KEY)
    batch_id = TOURNAMENT_BATCH_ID_FILE.read_text().strip()
    batch = client.beta.messages.batches.retrieve(
        batch_id, betas=["message-batches-2024-09-24"]
    )

    if batch.processing_status != "ended":
        print(f"Batch not ready yet. Status: {batch.processing_status}")
        return

    # Tally round wins per fight
    # round_votes[fight_key][category] = rounds won
    round_votes = defaultdict(lambda: defaultdict(int))
    errors = 0

    print(f"Downloading results for batch {batch_id}...")
    for result in client.beta.messages.batches.results(
        batch_id, betas=["message-batches-2024-09-24"]
    ):
        if result.result.type != "succeeded":
            errors += 1
            continue

        p = pairing_map[result.custom_id]
        raw = result.result.message.content[0].text.strip().rstrip(".").strip()

        fight_key = f"{p['cat_a']}||{p['cat_b']}||{p['fight_idx']}"

        if raw == "1":
            round_votes[fight_key][p["pos_1"]] += 1
        elif raw == "2":
            round_votes[fight_key][p["pos_2"]] += 1
        # If neither 1 nor 2, round is discarded (model didn't follow instructions)

    print(f"Results retrieved. Errors: {errors}")

    # Determine fight winner (needs >= 2 rounds out of 3)
    # matchup_wins[(cat_a, cat_b)][cat] = number of fights won
    matchup_wins = defaultdict(lambda: defaultdict(int))
    undecided = 0

    for fight_key, votes in round_votes.items():
        cat_a, cat_b, _ = fight_key.split("||")
        best_cat = max(votes, key=votes.get)
        best_score = votes[best_cat]

        if best_score >= 2:
            matchup_wins[(cat_a, cat_b)][best_cat] += 1
        else:
            undecided += 1

    print(f"Undecided fights (tie): {undecided}")

    # Merge both directions (A→B and B→A) into a single canonical matchup key
    merged_wins = defaultdict(lambda: defaultdict(int))
    for (cat_a, cat_b), wins in matchup_wins.items():
        key = tuple(sorted([cat_a, cat_b]))
        for cat, w in wins.items():
            merged_wins[key][cat] += w

    # Format results
    output = {}
    for (cat_a, cat_b), wins in merged_wins.items():
        total = sum(wins.values())
        output[f"{cat_a} vs {cat_b}"] = {
            cat: {
                "wins": w,
                "win_rate": round(w / total, 3) if total else 0,
            }
            for cat, w in sorted(wins.items(), key=lambda x: -x[1])
        }
        output[f"{cat_a} vs {cat_b}"]["total_fights"] = total

    with open(TOURNAMENT_RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {TOURNAMENT_RESULTS_FILE}")
    print(json.dumps(output, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "submit":
        submit()
    elif command == "status":
        status()
    elif command == "retrieve":
        retrieve()
    else:
        print(__doc__)
