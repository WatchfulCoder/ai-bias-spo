import pandas as pd

df = pd.read_csv(
    "jokes.tsv",
    sep="\t",
    header=None,
    names=["score", "joke"],
    on_bad_lines="skip",
    engine="python",
)

# Drop empty jokes and duplicates
df = df.dropna(subset=["joke"]).drop_duplicates(subset=["joke"])

COLUMNS = ["joke", "category_human", "category_llm"]

for i in range(1, 6):
    sample = df.sample(n=100, random_state=i)[["joke"]].reset_index(drop=True)
    sample.index += 1
    sample[["category_human", "category_llm"]] = ""
    sample[COLUMNS].to_csv(f"sample_{i}.csv", index_label="id")
    print(f"Sample {i} saved ({len(sample)} jokes)")
