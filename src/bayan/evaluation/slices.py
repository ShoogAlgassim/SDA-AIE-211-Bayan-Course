"""Lab 6: sliced evaluation report."""

import pandas as pd


def sliced_report(
    data,
    *,
    small_slice_threshold=30,
):
    """
    Compute sliced accuracy reports for:
    language, dialect, class, and length.

    Parameters
    ----------
    data:
        pandas DataFrame or path to a CSV file.

    small_slice_threshold:
        Slices with fewer than this number of examples
        are flagged as small.

    Returns
    -------
    dict
        Mapping from slice type to a list of metric rows.
    """

    if isinstance(data, (str, bytes)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    required = {
        "lang",
        "dialect_region",
        "length_bucket",
        "y_true",
        "y_pred",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["correct"] = (
        df["y_true"] == df["y_pred"]
    ).astype(int)

    slice_columns = {
        "language": "lang",
        "dialect": "dialect_region",
        "class": "y_true",
        "length": "length_bucket",
    }

    report = {}

    for slice_name, column in slice_columns.items():
        rows = []

        # Keep missing dialect values visible instead of dropping them.
        values = df[column].fillna("unknown")

        temp = df.copy()
        temp[column] = values

        for value, group in temp.groupby(column):
            n = len(group)

            accuracy = float(
                group["correct"].mean()
            )

            rows.append(
                {
                    "slice": str(value),
                    "n": int(n),
                    "accuracy": accuracy,
                    "small_slice": (
                        n < small_slice_threshold
                    ),
                }
            )

        rows.sort(
            key=lambda row: row["slice"]
        )

        report[slice_name] = rows

    return report