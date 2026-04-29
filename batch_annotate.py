"""
Batch annotation of 10,000 jokes using Anthropic's Message Batches API.

Usage:
    Step 1  Submit batch:   python batch_annotate.py submit
    Step 2  Check status:   python batch_annotate.py status
    Step 3  Retrieve results: python batch_annotate.py retrieve
"""

import sys
import anthropic
import pandas as pd
from typing import Any

from config import (
    ANNOTATED_JOKES_FILE,
    ANNOTATION_BATCH_ID_FILE,
    API_KEY,
    JOKES_10K_FILE,
    JOKES_FILE,
    MODEL,
    N_JOKES,
    N_SAMPLE_FILES,
)


# SYSTEM PROMPT


SYSTEM_PROMPT = """You are a content classifier. Your task is to classify jokes based on the TARGETED GROUP they refer to.

You must ONLY use the following categories:
1. Gender-based: jokes targeting gender roles or gender features.
2. Racial/Ethnic/National: jokes targeting race, ethnicity, or nationality.
3. Age-based: jokes targeting age groups.
4. Appearance/Physical condition: jokes targeting looks, body type, or physical disabilities.
5. Religion-based: jokes targeting religious groups or beliefs.
6. Class/Occupation: jokes targeting professions or socioeconomic class.
7. Sexual orientation: jokes targeting LGBTQ+ identities.
8. Political/Ideological: jokes targeting political figures, parties, or ideological groups.
9. Mental health: jokes targeting mental illness or psychological conditions.

Instructions:
- Only label a category if the joke meaningfully targets that group.
- A joke may belong to MULTIPLE categories, assign ALL that apply, separated by "; ".
- If no clear group is targeted, return exactly: None
- Do not rely only on keywords, use context and meaning.
- Return ONLY the category label(s), nothing else."""


# FEW-SHOT EXAMPLES
# Fill these in from your annotated samples.
# Rules: single-category, short, unambiguous.


FEW_SHOT_EXAMPLES = [
    # Gender-based
    {"joke": "Doctor: How is your headache Man: She is fine",          "category": "Gender-based"},
    {"joke": "When it’s April 2nd... And she’s still pregnant",          "category": "Gender-based"},
    # Racial/Ethnic/National
    {"joke": "Black Ninja 'I’m about to go incognegro'",          "category": "Racial/Ethnic/National"},
    {"joke": "What type of Asians do the most shopping? Shop-anese",          "category": "Racial/Ethnic/National"},
    # Age-based
    {"joke": "Why did the guitarist go to jail? He fingered a minor.",             "category": "Age-based"},
    {"joke": "I'd like to die like my grandpa, in my sleep. Not screaming and yelling, like the people in his car.",             "category": "Age-based"},
    # Appearance/Physical condition
    {"joke": "What std can you get from earrape? Hearing aids",      "category": "Appearance/Physical condition"},
    {"joke": "Helen Keller walks into a bar Then a table, then a chair.",      "category": "Appearance/Physical condition"},
    # Religion-based
    {"joke": "How does Moses make tea? Hebrews it.",        "category": "Religion-based"},
    {"joke": "What is a priest's cell phone provider? Virgin mobile",        "category": "Religion-based"},
    # Class/Occupation
    {"joke": "Why can't I differentiate between White Collar workers and Blue Collar workers? It's because I am Collarblind",           "category": "Class/Occupation"},
    {"joke": "how many programmer does it take to change a light bulb? 21, 1 person change it, 20 others to tell they can do better",           "category": "Class/Occupation"},
    # Sexual orientation
    {"joke": "What do you call being hit by a homosexual? Fruit punch",          "category": "Sexual orientation"},
    {"joke": "A boy brags to his friends that he just lost his virginity... A boy came running in a room full of his friends bragging about losing his virginity.  His friends asked him to sit down and tell them the whole story.  He replies I can't sit down, my ass hurts.", "category": "Sexual orientation"},
    # Political/Ideological
    {"joke": "What do you call someone whose filed bankruptcy 4 times and divorced twice? A GOP Presidential candidate.",       "category": "Political/Ideological"},
    {"joke": "If Kim Jong Un was a girl... Send nukes",       "category": "Political/Ideological"},
    # Mental health
    {"joke": "'Doctor, I have a drinking problem!' 'I'm always drunk whenever I'm traveling between countries!'  Doctor - 'Sounds like you're a borderline alcoholic.'",          "category": "Mental health"},
    {"joke": "I think I might be a polygamist ... ... My wife has multiple personality disorder.", "category": "Mental health"},
    # None
    {"joke": "What's the most powerful country after USA ?    USB.",            "category": "None"},
    {"joke": "A man proposed to a woman at the Gym but she said no. Its a shame that didn't workout. ",            "category": "None"},
]

def build_user_message(joke: str) -> str:
    """Build the classification prompt for a joke.

    Args:
        joke: Joke text to classify.

    Returns:
        The user message containing few-shot examples and the target joke.
    """
    lines = []
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f'Joke: "{ex["joke"]}"\nCategory: {ex["category"]}')
    lines.append(f'Joke: "{joke}"\nCategory:')
    return "\n\n".join(lines)



# STEP 1: LOAD & SAMPLE JOKES


def load_jokes() -> pd.DataFrame:
    """Load and sample jokes that have not already been annotated.

    Returns:
        A dataframe containing the sampled jokes for batch annotation.
    """
    full = pd.read_csv(
        JOKES_FILE,
        sep="\t",
        header=None,
        names=["score", "joke"],
        on_bad_lines="skip",
        engine="python",
    ).dropna(subset=["joke"]).drop_duplicates(subset=["joke"])

    already_sampled = set()
    for i in range(1, N_SAMPLE_FILES + 1):
        df = pd.read_csv(JOKES_FILE.parent / f"sample_{i}.csv")
        already_sampled.update(df["joke"].tolist())

    remaining = full[~full["joke"].isin(already_sampled)]
    sample = remaining.sample(n=N_JOKES, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample)} jokes from {len(remaining)} remaining.")
    return sample



# STEP 2: SUBMIT BATCH


def submit(jokes: pd.DataFrame) -> None:
    """Build and submit a joke annotation batch to Anthropic.

    Args:
        jokes: Dataframe of jokes to classify.

    Returns:
        None.
    """
    client = anthropic.Anthropic(api_key=API_KEY)

    requests: list[dict[str, Any]] = []
    for idx, row in jokes.iterrows():
        requests.append({
            "custom_id": f"joke_{idx}",
            "params": {
                "model": MODEL,
                "max_tokens": 50,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": build_user_message(row["joke"])}
                ],
            },
        })

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.beta.messages.batches.create(
        requests=requests,
        betas=["message-batches-2024-09-24"],
    )

    ANNOTATION_BATCH_ID_FILE.write_text(batch.id)
    jokes.to_csv(JOKES_10K_FILE, index_label="id")
    print(f"Batch submitted. ID: {batch.id}")
    print(f"Batch ID saved to {ANNOTATION_BATCH_ID_FILE}")
    print(f"Jokes saved to jokes_10k.csv")
    print(f"Status: {batch.processing_status}")



# STEP 3: CHECK STATUS


def status() -> None:
    """Check the processing status of the annotation batch.

    Returns:
        None.
    """
    if not ANNOTATION_BATCH_ID_FILE.exists():
        print("No batch ID found. Run 'submit' first.")
        return

    client = anthropic.Anthropic(api_key=API_KEY)
    batch_id = ANNOTATION_BATCH_ID_FILE.read_text().strip()
    batch = client.beta.messages.batches.retrieve(
        batch_id,
        betas=["message-batches-2024-09-24"],
    )

    print(f"Batch ID : {batch.id}")
    print(f"Status   : {batch.processing_status}")
    print(f"Counts   : {batch.request_counts}")
    if batch.processing_status == "ended":
        print("Batch is ready. Run 'retrieve' to get results.")



# STEP 4: RETRIEVE & SAVE RESULTS


def retrieve() -> None:
    """Download annotation results and merge them into a CSV.

    Returns:
        None.
    """
    if not ANNOTATION_BATCH_ID_FILE.exists():
        print("No batch ID found. Run 'submit' first.")
        return

    client = anthropic.Anthropic(api_key=API_KEY)
    batch_id = ANNOTATION_BATCH_ID_FILE.read_text().strip()
    batch = client.beta.messages.batches.retrieve(
        batch_id,
        betas=["message-batches-2024-09-24"],
    )

    if batch.processing_status != "ended":
        print(f"Batch not ready yet. Status: {batch.processing_status}")
        return

    jokes = pd.read_csv(JOKES_10K_FILE, index_col="id")
    jokes["category_llm"] = None

    print("Downloading results...")
    for result in client.beta.messages.batches.results(
        batch_id,
        betas=["message-batches-2024-09-24"],
    ):
        idx = int(result.custom_id.split("_")[1])
        if result.result.type == "succeeded":
            label = result.result.message.content[0].text.strip()
            jokes.at[idx, "category_llm"] = label
        else:
            jokes.at[idx, "category_llm"] = "ERROR"

    jokes.to_csv(ANNOTATED_JOKES_FILE, index_label="id")
    success = (jokes["category_llm"] != "ERROR").sum()
    print(f"Done. {success}/{len(jokes)} jokes annotated.")
    print(f"Results saved to {ANNOTATED_JOKES_FILE}")



# ENTRYPOINT


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "submit":
        jokes = load_jokes()
        submit(jokes)
    elif command == "status":
        status()
    elif command == "retrieve":
        retrieve()
    else:
        print(__doc__)
