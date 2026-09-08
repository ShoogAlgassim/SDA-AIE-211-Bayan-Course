"""Lab 4: fine-tune Bayan NER with Arabic clitic segmentation."""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from bayan.models.ner import align_labels
from bayan.preprocessing.arabic import segment


CHECKPOINT = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DATA_PATH = Path("data/models/bayan_ner.conll")
MAX_LENGTH = 128

LABELS = [
    "O",
    "B-SERVICE",
    "B-LOCATION",
    "B-DATE",
    "B-REFERENCE",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def parse_conll(path):
    sentences = []
    words = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()

            if not line:
                if words:
                    sentences.append(
                        {
                            "words": words,
                            "labels": labels,
                        }
                    )
                    words = []
                    labels = []
                continue

            parts = line.split()

            label = parts[-1]
            token = " ".join(parts[:-1])

            words.append(token)
            labels.append(label)

    if words:
        sentences.append(
            {
                "words": words,
                "labels": labels,
            }
        )

    return sentences


class NERDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        use_segmentation=False,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.use_segmentation = use_segmentation

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        words = example["words"]
        labels = example["labels"]

        # Lab 4: Arabic clitic segmentation
        if self.use_segmentation:
            segmented_words = []
            segmented_word_labels = []

            for word, label in zip(words, labels):
                pieces = segment(word)

                if not pieces:
                    pieces = [word]

                for piece_index, piece in enumerate(pieces):
                    segmented_words.append(piece)

                    # Only the first clitic piece receives
                    # the original word-level NER label.
                    if piece_index == 0:
                        segmented_word_labels.append(
                            LABEL2ID[label]
                        )
                    else:
                        # Ignore continuation clitic pieces
                        # during loss/evaluation.
                        segmented_word_labels.append(-100)

            words = segmented_words
            word_labels = segmented_word_labels

        else:
            word_labels = [
                LABEL2ID[label]
                for label in labels
            ]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(
            batch_index=0
        )

        # align_labels also masks tokenizer subword
        # continuation pieces with -100.
        aligned_labels = align_labels(
            word_ids,
            word_labels,
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            aligned_labels,
            dtype=torch.long,
        )

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    true_predictions = []
    true_labels = []

    for prediction, label_ids in zip(
        predictions,
        labels,
    ):
        sentence_predictions = []
        sentence_labels = []

        for pred_id, label_id in zip(
            prediction,
            label_ids,
        ):
            if label_id == -100:
                continue

            sentence_predictions.append(
                ID2LABEL[int(pred_id)]
            )

            sentence_labels.append(
                ID2LABEL[int(label_id)]
            )

        true_predictions.append(
            sentence_predictions
        )

        true_labels.append(
            sentence_labels
        )

    report = classification_report(
        true_labels,
        true_predictions,
        output_dict=True,
        zero_division=0,
    )

    location_recall = (
        report.get("LOCATION", {})
        .get("recall", 0.0)
    )

    return {
        "precision": precision_score(
            true_labels,
            true_predictions,
        ),
        "recall": recall_score(
            true_labels,
            true_predictions,
        ),
        "f1": f1_score(
            true_labels,
            true_predictions,
        ),
        "accuracy": accuracy_score(
            true_labels,
            true_predictions,
        ),
        "location_recall": location_recall,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="artifacts/ner_segmented",
        help="Where to save the segmented NER artefact.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Parse CoNLL data
    data = parse_conll(DATA_PATH)

    print("Total sentences:", len(data))

    # 2. Same deterministic split as Lab 3B
    train_data, temp_data = train_test_split(
        data,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    validation_data, test_data = train_test_split(
        temp_data,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    print("Train size:", len(train_data))
    print("Validation size:", len(validation_data))
    print("Test size:", len(test_data))

    # 3. Load tokenizer
    print("Loading checkpoint:", CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT
    )

    # 4. Build datasets WITH clitic segmentation
    train_dataset = NERDataset(
        train_data,
        tokenizer,
        use_segmentation=True,
    )

    validation_dataset = NERDataset(
        validation_data,
        tokenizer,
        use_segmentation=True,
    )

    test_dataset = NERDataset(
        test_data,
        tokenizer,
        use_segmentation=True,
    )

    # 5. Load token classification model
    model = AutoModelForTokenClassification.from_pretrained(
        CHECKPOINT,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    # 6. Training configuration
    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=3,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,

        learning_rate=2e-5,
        weight_decay=0.01,

        evaluation_strategy="epoch",
        save_strategy="epoch",

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        save_total_limit=1,
        save_safetensors=False,

        fp16=torch.cuda.is_available(),

        logging_steps=50,
        report_to="none",

        seed=42,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        tokenizer=tokenizer,

        compute_metrics=compute_metrics,
    )

    # 8. Train
    print("\nStarting segmented NER training...")

    trainer.train()

    # 9. Validation
    print("\nValidation metrics:")

    validation_metrics = trainer.evaluate(
        validation_dataset
    )

    print(validation_metrics)

    # 10. Frozen test
    print("\nFrozen test metrics:")

    test_metrics = trainer.evaluate(
        test_dataset,
        metric_key_prefix="test",
    )

    print(test_metrics)

    print(
        "\nSegmented LOCATION recall:",
        test_metrics.get("test_location_recall"),
    )

    # 11. Save model + tokenizer
    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print("\nSaved segmented NER model and tokenizer to:")
    print(output_dir)


if __name__ == "__main__":
    main()