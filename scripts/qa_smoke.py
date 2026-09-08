"""Lab 3B: run the QA smoke set."""

import json

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from bayan.models.qa import best_span


CHECKPOINT = "deepset/roberta-base-squad2"
SMOKE_PATH = "data/eval/qa_smoke_set.json"


def load_smoke_set(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []

    for item in data["data"]:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "context": context,
                        "answers": qa["answers"],
                        "is_impossible": qa["is_impossible"],
                    }
                )

    return examples


def main():
    print("Loading QA checkpoint:", CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT,
        use_fast=True,
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        CHECKPOINT
    )

    model.eval()

    examples = load_smoke_set(SMOKE_PATH)

    answerable_correct = 0
    answerable_total = 0

    null_correct = 0
    null_total = 0

    for example in examples:
        question = example["question"]
        context = example["context"]

        encoding = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=384,
        )

        offsets = encoding.pop("offset_mapping")[0].tolist()

        sequence_ids = encoding.sequence_ids(0)

        context_offsets = []

        for offset, sequence_id in zip(
            offsets,
            sequence_ids,
        ):
            if sequence_id == 1:
                context_offsets.append(
                    tuple(offset)
                )
            else:
                context_offsets.append(None)

        with torch.no_grad():
            outputs = model(**encoding)

        start_logits = (
            outputs.start_logits[0]
            .cpu()
            .numpy()
        )

        end_logits = (
            outputs.end_logits[0]
            .cpu()
            .numpy()
        )

        null_score = float(
            start_logits[0] + end_logits[0]
        )

        result = best_span(
            start_logits,
            end_logits,
            context_offsets,
            null_score=null_score,
            null_threshold=0.0,
            max_answer_len=30,
            top_k=20,
        )

        predicted_answer = None

        if result["answer"] is not None:
            start_char, end_char = result["answer"]

            predicted_answer = context[
                start_char:end_char
            ]

        if example["is_impossible"]:
            null_total += 1

            if predicted_answer is None:
                null_correct += 1

            print(
                example["id"],
                "| expected=None",
                "| predicted=",
                predicted_answer,
            )

        else:
            answerable_total += 1

            gold_answers = [
                answer["text"]
                for answer in example["answers"]
            ]

            is_correct = (
                predicted_answer in gold_answers
            )

            if is_correct:
                answerable_correct += 1

            print(
                example["id"],
                "| expected=",
                gold_answers,
                "| predicted=",
                predicted_answer,
                "| correct=",
                is_correct,
            )

    print("\nQA Smoke Summary")
    print(
        f"Answerable: "
        f"{answerable_correct}/{answerable_total}"
    )

    print(
        f"Unanswerable: "
        f"{null_correct}/{null_total}"
    )


if __name__ == "__main__":
    main()