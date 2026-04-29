import pandas as pd

from config import JOKES_FILE, N_SAMPLE_FILES, SAMPLE_COLUMNS, SAMPLE_SIZE

df = pd.read_csv(
    JOKES_FILE,
    sep="\t",
    header=None,
    names=["score", "joke"],
    on_bad_lines="skip",
    engine="python",
)

# Drop empty jokes and duplicates
df = df.dropna(subset=["joke"]).drop_duplicates(subset=["joke"])

for i in range(1, N_SAMPLE_FILES + 1):
    sample = df.sample(n=SAMPLE_SIZE, random_state=i)[["joke"]].reset_index(drop=True)
    sample.index += 1
    sample[["category_human", "category_llm"]] = ""
    sample[SAMPLE_COLUMNS].to_csv(JOKES_FILE.parent / f"sample_{i}.csv", index_label="id")
    print(f"Sample {i} saved ({len(sample)} jokes)")
