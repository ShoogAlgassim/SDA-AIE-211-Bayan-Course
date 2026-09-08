"""Lab 3A: fine-tune the Bayan topic classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from bayan.models.data import build_topic_dataset


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
MAX_LENGTH = 256


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

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/topic_classifier",
        help="Where to save the trained classifier artefact.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Load grouped dataset
    ds = build_topic_dataset()

    train_df = ds["train"]
    validation_df = ds["validation"]
    test_df = ds["test"]

    print("Train size:", len(train_df))
    print("Validation size:", len(validation_df))
    print("Test size:", len(test_df))

    # 2. Build label mappings
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

    print("Labels:", label2id)

    # 3. Load Lab-1 checkpoint and matching tokenizer
    print("Loading checkpoint:", CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )

    # 4. Tokenize datasets
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

    test_dataset = TopicDataset(
        test_df,
        tokenizer,
        label2id,
    )

    # 5. Training configuration
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

        # Avoid safetensors non-contiguous tensor saving error
        save_safetensors=False,

        fp16=torch.cuda.is_available(),

        logging_steps=50,
        report_to="none",

        seed=42,
    )

    # 6. Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics,
    )

    # 7. Fine-tune
    print("\nStarting training...")

    trainer.train()

    # 8. Validation evaluation
    print("\nValidation metrics:")

    validation_metrics = trainer.evaluate(
        validation_dataset
    )

    print(validation_metrics)

    # 9. Frozen test evaluation
    print("\nFrozen test metrics:")

    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    print(test_metrics)

    # 10. Save re-runnable artefact
    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print("\nSaved model and tokenizer to:")
    print(output_dir)


if __name__ == "__main__":
    main()