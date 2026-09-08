"""Lab 3 starter: dataset construction and split integrity."""

from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def build_topic_dataset():
    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"].reset_index(drop=True)
    validation = df[df["split"] == "validation"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }