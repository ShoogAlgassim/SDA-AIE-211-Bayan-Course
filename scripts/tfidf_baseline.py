"""Lab 3A: TF-IDF + LinearSVC baseline."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bayan.models.data import build_topic_dataset


def main():
    # 1. Load the grouped dataset
    ds = build_topic_dataset()

    train_df = ds["train"]
    test_df = ds["test"]

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    # 2. Build TF-IDF + LinearSVC baseline
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )

    # 3. Train
    model.fit(train_df["text"], train_df["topic"])

    # 4. Predict on frozen test split
    predictions = model.predict(test_df["text"])

    # 5. Compute macro-F1
    macro_f1 = f1_score(
        test_df["topic"],
        predictions,
        average="macro",
    )

    print(f"Baseline macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()