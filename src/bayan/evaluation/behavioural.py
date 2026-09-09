"""Lab 6: behavioural test generators/runners."""

from pathlib import Path

import pandas as pd


DEFAULT_TEMPLATE_PATH = Path(
    "data/eval/behavioural_templates.csv"
)


def _safe_rate(passed, total):
    if total == 0:
        return None

    return passed / total


def run_behavioural_suite(
    *,
    topic_predictor=None,
    sentiment_predictor=None,
    mft_cases=None,
    template_path=DEFAULT_TEMPLATE_PATH,
):
    """
    Run Bayan behavioural tests.

    Parameters
    ----------
    topic_predictor:
        Callable that takes a string and returns a topic label.

    sentiment_predictor:
        Callable that takes a string and returns a numeric sentiment score.
        Higher values are treated as more positive sentiment.

    mft_cases:
        Optional iterable of dictionaries with:
            text
            expected
        MFT cases are evaluated with topic_predictor.

    template_path:
        CSV containing the supplied invariance/directional templates.

    Returns
    -------
    dict
        Behavioural pass counts and rates.
    """

    df = pd.read_csv(template_path)

    required = {
        "test_id",
        "test_type",
        "lang",
        "template",
        "term",
        "expected_relation",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing behavioural columns: {sorted(missing)}"
        )

    results = {
        "invariance": {
            "passed": 0,
            "total": 0,
            "rate": None,
        },
        "directional": {
            "passed": 0,
            "total": 0,
            "rate": None,
        },
        "mft": {
            "passed": 0,
            "total": 0,
            "rate": None,
        },
    }

    invariance_rows = df[
        df["test_type"] == "invariance"
    ]

    if topic_predictor is not None:
        for _, row in invariance_rows.iterrows():
            template = str(row["template"])

            original_text = template.format(
                term=""
            ).strip()

            changed_text = template.format(
                term=str(row["term"])
            )

            original_prediction = topic_predictor(
                original_text
            )

            changed_prediction = topic_predictor(
                changed_text
            )

            passed = (
                original_prediction
                == changed_prediction
            )

            results["invariance"]["total"] += 1

            if passed:
                results["invariance"]["passed"] += 1

    directional_rows = df[
        df["test_type"] == "directional"
    ]

    if sentiment_predictor is not None:
        for _, row in directional_rows.iterrows():
            template = str(row["template"])

            changed_text = template.format(
                term=str(row["term"])
            )

            positive_text = (
                changed_text
                .replace(" not ", " ")
                .replace("لا ", "")
            )

            positive_score = float(
                sentiment_predictor(
                    positive_text
                )
            )

            negative_score = float(
                sentiment_predictor(
                    changed_text
                )
            )

            passed = (
                negative_score
                <= positive_score
            )

            results["directional"]["total"] += 1

            if passed:
                results["directional"]["passed"] += 1

    if (
        topic_predictor is not None
        and mft_cases is not None
    ):
        for case in mft_cases:
            text = case["text"]
            expected = case["expected"]

            prediction = topic_predictor(text)

            passed = (
                prediction == expected
            )

            results["mft"]["total"] += 1

            if passed:
                results["mft"]["passed"] += 1

    for test_type in results:
        results[test_type]["rate"] = _safe_rate(
            results[test_type]["passed"],
            results[test_type]["total"],
        )

    return results
