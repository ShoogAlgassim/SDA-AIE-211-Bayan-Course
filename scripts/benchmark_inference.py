"""Lab 7: CPU inference benchmark for PyTorch and ONNX models."""

import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = Path(
    "/content/drive/MyDrive/SDA-AIE-211/artifacts/topic_classifier"
)

BENCH_MIX_PATH = Path(
    "data/serving/bench_mix.npy"
)

FP32_ONNX_PATH = Path(
    "artifacts/lab7_classifier/classifier_fp32.onnx"
)

INT8_ONNX_PATH = Path(
    "artifacts/lab7_classifier/classifier_int8.onnx"
)

THREADS = 4
WARMUP_RUNS = 10
MEASURE_RUNS = 100
MAX_LENGTH = 128


def load_length_mix():
    return np.load(
        BENCH_MIX_PATH,
        allow_pickle=True,
    )


def make_text(item):
    if isinstance(item, str):
        return item

    length = int(item)

    base = "الخدمة تحتاج متابعة "
    return (
        base
        * max(1, length // 3 + 1)
    ).strip()


def percentile_result(latencies):
    return {
        "p50_ms": float(
            np.percentile(latencies, 50)
        ),
        "p99_ms": float(
            np.percentile(latencies, 99)
        ),
    }


def benchmark_pytorch(
    model,
    tokenizer,
    texts,
):
    torch.set_num_threads(THREADS)

    os.environ["OMP_NUM_THREADS"] = str(THREADS)

    def run(text):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.inference_mode():
            _ = model(**encoded)

    for i in range(WARMUP_RUNS):
        run(texts[i % len(texts)])

    latencies = []

    for i in range(MEASURE_RUNS):
        text = texts[i % len(texts)]

        start = time.perf_counter()
        run(text)
        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    return percentile_result(latencies)


def make_ort_session(path):
    options = ort.SessionOptions()

    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_onnx(
    session,
    tokenizer,
    texts,
):
    input_names = {
        inp.name
        for inp in session.get_inputs()
    }

    def run(text):
        encoded = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        inputs = {}

        if "input_ids" in input_names:
            inputs["input_ids"] = (
                encoded["input_ids"].astype(
                    np.int64
                )
            )

        if "attention_mask" in input_names:
            inputs["attention_mask"] = (
                encoded["attention_mask"].astype(
                    np.int64
                )
            )

        if (
            "token_type_ids" in input_names
            and "token_type_ids" in encoded
        ):
            inputs["token_type_ids"] = (
                encoded["token_type_ids"].astype(
                    np.int64
                )
            )

        session.run(
            None,
            inputs,
        )

    for i in range(WARMUP_RUNS):
        run(texts[i % len(texts)])

    latencies = []

    for i in range(MEASURE_RUNS):
        text = texts[i % len(texts)]

        start = time.perf_counter()
        run(text)
        end = time.perf_counter()

        latencies.append(
            (end - start) * 1000
        )

    return percentile_result(latencies)


def print_row(name, result):
    print(
        f"{name} | "
        f"p50={result['p50_ms']:.2f} ms | "
        f"p99={result['p99_ms']:.2f} ms"
    )


def benchmark():
    print("Pinned CPU threads:", THREADS)
    print("max_length:", MAX_LENGTH)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    mix = load_length_mix()

    texts = [
        make_text(item)
        for item in mix
    ]

    print(
        "Production mix examples:",
        len(texts),
    )

    # ---------------------------------
    # PyTorch FP32
    # ---------------------------------
    print("\nLoading PyTorch FP32...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_PATH)
    )

    model.eval()
    model.to("cpu")

    pytorch_result = benchmark_pytorch(
        model,
        tokenizer,
        texts,
    )

    print_row(
        "PyTorch FP32 dynamic 128",
        pytorch_result,
    )

    # ---------------------------------
    # ONNX FP32
    # ---------------------------------
    print("\nLoading ONNX FP32...")

    fp32_session = make_ort_session(
        FP32_ONNX_PATH
    )

    onnx_fp32_result = benchmark_onnx(
        fp32_session,
        tokenizer,
        texts,
    )

    print_row(
        "ONNX FP32 dynamic 128",
        onnx_fp32_result,
    )

    # ---------------------------------
    # ONNX INT8
    # ---------------------------------
    print("\nLoading ONNX INT8...")

    int8_session = make_ort_session(
        INT8_ONNX_PATH
    )

    int8_result = benchmark_onnx(
        int8_session,
        tokenizer,
        texts,
    )

    print_row(
        "ONNX INT8 dynamic 128",
        int8_result,
    )

    # ---------------------------------
    # Speed-ups
    # ---------------------------------
    fp32_speedup = (
        pytorch_result["p99_ms"]
        / onnx_fp32_result["p99_ms"]
    )

    int8_speedup = (
        pytorch_result["p99_ms"]
        / int8_result["p99_ms"]
    )

    print("\n=== Speed-up ===")

    print(
        f"ONNX FP32 p99 speed-up: "
        f"{fp32_speedup:.2f}x"
    )

    print(
        f"ONNX INT8 p99 speed-up: "
        f"{int8_speedup:.2f}x"
    )

    return {
        "pytorch_fp32": pytorch_result,
        "onnx_fp32": onnx_fp32_result,
        "onnx_int8": int8_result,
        "onnx_fp32_speedup": fp32_speedup,
        "onnx_int8_speedup": int8_speedup,
    }


if __name__ == "__main__":
    benchmark()
