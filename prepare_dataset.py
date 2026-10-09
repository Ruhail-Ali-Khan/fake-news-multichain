
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


def clean_text(text):
    text = str(text)

    # Remove common source dateline markers
    text = re.sub(
        r"^.{0,100}?\(Reuters\)\s*[-–—]\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bReuters\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def prepare_dataset():
    fake = pd.read_csv(RAW_DIR / "Fake.csv")
    real = pd.read_csv(RAW_DIR / "True.csv")

    required = {"title", "text"}

    if not required.issubset(fake.columns):
        raise ValueError("Fake.csv missing required columns")

    if not required.issubset(real.columns):
        raise ValueError("True.csv missing required columns")

    fake["label"] = 0
    real["label"] = 1

    print("Fake news records:", len(fake))
    print("Real news records:", len(real))

    data = pd.concat([fake, real], ignore_index=True)

    data["title"] = data["title"].fillna("")
    data["text"] = data["text"].fillna("")

    data["text_clean"] = (
        data["title"].map(clean_text)
        + " "
        + data["text"].map(clean_text)
    ).str.strip()

    # Remove empty/very short articles
    data = data[data["text_clean"].str.len() >= 80].copy()

    # Remove articles with contradictory labels
    conflicting = (
        data.groupby("text_clean")["label"]
        .transform("nunique") > 1
    )
    data = data[~conflicting].copy()

    # Remove exact cleaned-text duplicates
    data = data.drop_duplicates(subset=["text_clean"])

    # Reproducible shuffling
    data = data.sample(frac=1, random_state=42)
    data = data.reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "news_dataset.csv"

    data[["text_clean", "label"]].to_csv(
        output_path,
        index=False
    )

    print("\nDATASET PREPARATION COMPLETED")
    print("-----------------------------")
    print("Final records:", len(data))
    print("Fake records:", (data["label"] == 0).sum())
    print("Real records:", (data["label"] == 1).sum())
    print("Saved at:", output_path)


if __name__ == "__main__":
    prepare_dataset()
