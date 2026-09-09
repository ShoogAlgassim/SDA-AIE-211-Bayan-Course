"""Lab 7: ONNX export and dynamic INT8 quantisation."""

import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)


MODEL_PATH = Path(
    "/content/drive/MyDrive/SDA-AIE-211/artifacts/topic_classifier"
)

OUTPUT_DIR = Path("artifacts/lab7_classifier")

FP32_ONNX_PATH = OUTPUT_DIR / "classifier_fp32.onnx"
INT8_ONNX_PATH = OUTPUT_DIR / "classifier_int8.onnx"
ROLLBACK_DIR = OUTPUT_DIR / "rollback_fp32"


def export_classifier():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading classifier...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()
    model.to("cpu")

    # -----------------------------------
    # Preserve rollback FP32 artefact
    # -----------------------------------
    if ROLLBACK_DIR.exists():
        shutil.rmtree(ROLLBACK_DIR)

    shutil.copytree(
        MODEL_PATH,
        ROLLBACK_DIR,
    )

    print(
        "Rollback FP32 artefact saved:",
        ROLLBACK_DIR,
    )

    # -----------------------------------
    # Dummy export input
    # -----------------------------------
    sample = tokenizer(
        "الخدمة تحتاج متابعة",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    # -----------------------------------
    # Export FP32 ONNX
    # -----------------------------------
    print("Exporting classifier to ONNX FP32...")

    torch.onnx.export(
        model,
        (
            input_ids,
            attention_mask,
        ),
        str(FP32_ONNX_PATH),
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_length",
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length",
            },
            "logits": {
                0: "batch_size",
            },
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    print(
        "FP32 ONNX saved:",
        FP32_ONNX_PATH,
    )

    # -----------------------------------
    # INT8 dynamic quantisation
    # -----------------------------------
    print("Quantising classifier to INT8...")

    quantize_dynamic(
        model_input=str(FP32_ONNX_PATH),
        model_output=str(INT8_ONNX_PATH),
        weight_type=QuantType.QInt8,
    )

    print(
        "INT8 ONNX saved:",
        INT8_ONNX_PATH,
    )


def main():
    export_classifier()

    print("\nLab 7 classifier export complete.")
    print("FP32:", FP32_ONNX_PATH)
    print("INT8:", INT8_ONNX_PATH)
    print("Rollback:", ROLLBACK_DIR)


if __name__ == "__main__":
    main()