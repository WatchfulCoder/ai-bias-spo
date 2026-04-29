# AI Bias in Humor — Sciences Po M1 Research Project

This project investigates whether large language models exhibit systematic biases when judging humor. Specifically, we ask: does Claude rate jokes differently depending on the social group they target? Using a dataset of ~43,000 Reddit jokes, we annotate a 10,000-joke sample by targeted group, then run a large-scale tournament where Claude repeatedly judges which of two jokes is funnier. The results reveal clear and statistically significant preferences that reflect broader patterns of social bias encoded in LLMs.

---

## Dataset

| File | Description |
|------|-------------|
| `jokes.tsv` | Raw dataset of ~43,000 Reddit jokes with Reddit upvote scores (two columns: `score`, `joke`) |
| `sample_1.csv` … `sample_5.csv` | Five samples of 100 jokes each, used for initial human annotation. Each file contains columns `joke`, `category_human`, and `category_llm` |
| `annotation.xlsx` | Excel workbook with one tab per sample, used during the double-blind human annotation phase |
| `jokes_10k.csv` | The 10,000 jokes sampled for LLM annotation (excludes the 500 already in the samples above) |
| `annotated_10k.csv` | The 10,000 jokes with their LLM-assigned category labels (`category_llm`). Available on external storage — see link below |

> **Large data files** (`jokes_10k.csv`, `annotated_10k.csv`) are hosted externally due to GitHub size limitations:
> 📁 [Google Drive — annotated dataset](_LINK_TO_BE_ADDED_)

---

## Annotation Pipeline

| File | Description |
|------|-------------|
| `sample_jokes.py` | Samples 5 × 100 jokes from `jokes.tsv` and generates `sample_1.csv` … `sample_5.csv` |
| `annotate.py` | Hardcodes the human annotations for all 500 sample jokes and writes them back to the CSV files |
| `batch_annotate.py` | Annotates 10,000 jokes using the Anthropic Batch API (`submit` / `status` / `retrieve` commands). Uses few-shot prompting across 9 categories: Gender-based, Racial/Ethnic/National, Age-based, Appearance/Physical condition, Religion-based, Class/Occupation, Sexual orientation, Political/Ideological, Mental health |
| `batch_id.txt` | Stores the Anthropic batch ID for the annotation job |

---

## Tournament

| File | Description |
|------|-------------|
| `tournament.py` | Runs the joke tournament using the Anthropic Batch API (`submit` / `status` / `retrieve` commands). Implements a symmetric Design B: all ordered category pairs, K=3 opponents per joke, 3 rounds per fight (majority wins). ~99,800 API calls total |
| `tournament_batch_id.txt` | Stores the Anthropic batch ID for the tournament job |
| `tournament_pairings.json` | Maps each batch request ID to its metadata (categories, fight index, round index, joke texts, positions). Regenerated deterministically via `regenerate_pairings.py` — not committed due to size |
| `regenerate_pairings.py` | Reconstructs `tournament_pairings.json` from scratch without any API call, using the same random seed (42) as the original submission |
| `tournament_results.json` | Aggregated results: for each of the 36 category matchups, the number of fight wins and win rate per category |
| `joke_winrates.json` | Per-joke results: for every individual joke in the tournament, its win rate, number of fights, and category |

---

## Analysis & Visualizations

| File | Description |
|------|-------------|
| `analysis.py` | Statistical analysis: global win rates with 95% confidence intervals, chi-squared tests per matchup, and position bias analysis (re-streams raw batch results from the API) |
| `visualize.py` | Generates three figures: win rate ranking bar chart, 9×9 head-to-head heatmap, and position bias per category |
| `length_control.py` | Robustness check: tests whether joke length (characters, words) explains the tournament results. Computes Pearson correlation and Kruskal-Wallis test |
| `joke_winrates.py` | Computes per-joke win rates by re-streaming the batch results, saves `joke_winrates.json`, and generates a per-joke scatter plot |
| `length_scatter.py` | Scatter plot of individual joke win rate vs. joke length, color-coded by category |

---

## Figures

All figures are in `docs/figures/` and embedded in the project website.

| File | Description |
|------|-------------|
| `winrate_ranking.png` | Horizontal bar chart of global win rates per category with 95% CI |
| `matchup_heatmap.png` | 9×9 heatmap of head-to-head win rates between all category pairs |
| `position_bias.png` | Win rates in position 1 vs. position 2 per category |
| `length_control.png` | Joke length distributions and correlation with win rate at category level |
| `joke_scatter.png` | Per-joke win rates, one dot per joke, grouped by category |
| `length_winrate_scatter.png` | Individual joke win rate vs. joke length, color-coded by category |

---

## Key Findings

- **Category bias**: Claude systematically rates Class/Occupation jokes as funniest (66% win rate) and Sexual orientation and Racial/Ethnic jokes as least funny (~38%). All results are statistically significant (p < 10⁻⁸²).
- **Recency bias**: Claude chooses the joke presented in second position 65% of the time regardless of content, revealing a strong recency effect in pairwise LLM judgment.
- **Interaction effect**: Strong categories (Class, Religion) maintain above-50% win rates even when presented in the disadvantaged first position. Weak categories (Sexual orientation, Racial) fall to ~20% in first position, meaning they only stay competitive thanks to the position advantage.
- **Length as partial confound**: Joke length correlates positively with individual win rates (r=+0.35, p<0.001), but does not fully account for the category hierarchy — Class/Occupation dominates despite having the 5th shortest mean joke length.

---

## Website

Full results, methodology and interactive figures:
https://watchfulcoder.github.io/ai-bias-spo/
