"""Lab 4: compare Arabic-centric checkpoints on all/Gulf/MSA slices."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


DATA_PATH = Path("data/raw/bayan_feedback.csv")

MODELS = {
    "CAMeLBERT-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "CAMeLBERT-DA": "CAMeL-Lab/bert-base-arabic-camelbert-da",
}

MAX_LENGTH = 256
OUTPUT_ROOT = Path("artifacts/arabic_bakeoff")


class TopicDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id):
        self.labels = [
            label2id[label]
            for label in dataframe["topic"].tolist()
        ]

        self.encodings = tokenizer(
            dataframe["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
    }


def build_arabic_grouped_split():
    df = pd.read_csv(DATA_PATH)

    # Arabic slice only
    arabic_df = df[
        df["lang"] == "ar"
    ].reset_index(drop=True)

    # 70% train, 30% temporary
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, temp_idx = next(
        first_split.split(
            arabic_df,
            groups=arabic_df["citizen_group_id"],
        )
    )

    train_df = arabic_df.iloc[
        train_idx
    ].reset_index(drop=True)

    temp_df = arabic_df.iloc[
        temp_idx
    ].reset_index(drop=True)

    # Split remaining 30% equally:
    # 15% validation, 15% test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    validation_idx, test_idx = next(
        second_split.split(
            temp_df,
            groups=temp_df["citizen_group_id"],
        )
    )

    validation_df = temp_df.iloc[
        validation_idx
    ].reset_index(drop=True)

    test_df = temp_df.iloc[
        test_idx
    ].reset_index(drop=True)

    return train_df, validation_df, test_df


def check_group_overlap(
    train_df,
    validation_df,
    test_df,
):
    train_ids = set(
        train_df["citizen_group_id"]
    )
    validation_ids = set(
        validation_df["citizen_group_id"]
    )
    test_ids = set(
        test_df["citizen_group_id"]
    )

    print("\nCitizen overlap:")
    print(
        "Train / Validation:",
        len(train_ids & validation_ids),
    )
    print(
        "Train / Test:",
        len(train_ids & test_ids),
    )
    print(
        "Validation / Test:",
        len(validation_ids & test_ids),
    )

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def evaluate_slice(
    trainer,
    dataframe,
    tokenizer,
    label2id,
    name,
):
    dataset = TopicDataset(
        dataframe,
        tokenizer,
        label2id,
    )

    metrics = trainer.evaluate(
        dataset,
        metric_key_prefix=name,
    )

    return {
        "macro_f1": metrics[
            f"{name}_macro_f1"
        ],
        "accuracy": metrics[
            f"{name}_accuracy"
        ],
    }


def run_model(
    model_name,
    checkpoint,
    train_df,
    validation_df,
    test_df,
    label2id,
    id2label,
):
    print("\n" + "=" * 60)
    print("Model:", model_name)
    print("Checkpoint:", checkpoint)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    train_dataset = TopicDataset(
        train_df,
        tokenizer,
        label2id,
    )

    validation_dataset = TopicDataset(
        validation_df,
        tokenizer,
        label2id,
    )

    output_dir = OUTPUT_ROOT / model_name

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=2,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        learning_rate=2e-5,
        weight_decay=0.01,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        save_total_limit=1,
        save_safetensors=False,

        fp16=torch.cuda.is_available(),

        logging_steps=50,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    trainer.train()

    all_arabic = test_df.copy()

    gulf = test_df[
        test_df["dialect_region"] == "Gulf"
    ].reset_index(drop=True)

    msa = test_df[
        test_df["dialect_region"] == "MSA"
    ].reset_index(drop=True)

    print("\nSlice sizes:")
    print("All Arabic:", len(all_arabic))
    print("Gulf:", len(gulf))
    print("MSA:", len(msa))

    results = {
        "all": evaluate_slice(
            trainer,
            all_arabic,
            tokenizer,
            label2id,
            "all",
        ),
        "gulf": evaluate_slice(
            trainer,
            gulf,
            tokenizer,
            label2id,
            "gulf",
        ),
        "msa": evaluate_slice(
            trainer,
            msa,
            tokenizer,
            label2id,
            "msa",
        ),
    }

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    return results


def main():
    train_df, validation_df, test_df = (
        build_arabic_grouped_split()
    )

    print("Arabic train:", len(train_df))
    print(
        "Arabic validation:",
        len(validation_df),
    )
    print("Arabic test:", len(test_df))

    check_group_overlap(
        train_df,
        validation_df,
        test_df,
    )

    print("\nTest dialect distribution:")
    print(
        test_df[
            "dialect_region"
        ].value_counts()
    )

    labels = sorted(
        train_df["topic"].unique()
    )

    label2id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    all_results = {}

    for model_name, checkpoint in MODELS.items():
        all_results[model_name] = run_model(
            model_name,
            checkpoint,
            train_df,
            validation_df,
            test_df,
            label2id,
            id2label,
        )

    print(
        "\n\n===== ARABIC MODEL BAKE-OFF ====="
    )

    for model_name, results in all_results.items():
        print(f"\n{model_name}")

        print(
            "All Macro-F1:",
            f"{results['all']['macro_f1']:.4f}",
        )

        print(
            "Gulf Macro-F1:",
            f"{results['gulf']['macro_f1']:.4f}",
        )

        print(
            "MSA Macro-F1:",
            f"{results['msa']['macro_f1']:.4f}",
        )

    mix_gulf = all_results[
        "CAMeLBERT-mix"
    ]["gulf"]["macro_f1"]

    da_gulf = all_results[
        "CAMeLBERT-DA"
    ]["gulf"]["macro_f1"]

    delta = da_gulf - mix_gulf

    print("\nGulf slice comparison:")
    print(
        "CAMeLBERT-mix:",
        f"{mix_gulf:.4f}",
    )
    print(
        "CAMeLBERT-DA:",
        f"{da_gulf:.4f}",
    )
    print(
        "DA vs Mix delta:",
        f"{delta:+.4f}",
    )


if __name__ == "__main__":
    main()