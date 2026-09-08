"""Lab 4: audit dialect mix over the Arabic slice."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/bayan_feedback.csv")


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep Arabic feedback only
    arabic_df = df[df["lang"] == "ar"].copy()

    print("Arabic rows:", len(arabic_df))

    print("\nDialect / region distribution:")
    counts = arabic_df["dialect_region"].value_counts(dropna=False)

    percentages = (
        arabic_df["dialect_region"]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    for region, count in counts.items():
        pct = percentages.loc[region]
        print(f"{region}: {count} ({pct:.2f}%)")


if __name__ == "__main__":
    main()