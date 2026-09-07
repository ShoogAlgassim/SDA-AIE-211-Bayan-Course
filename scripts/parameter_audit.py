"""Lab 2 starter: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    """Count model parameters by subsystem."""

    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        count = param.numel()

        lname = name.lower()

        if "embeddings" in lname:
            buckets["embeddings"] += count

        elif "attention" in lname:
            buckets["attention"] += count

        elif "intermediate" in lname or "output.dense" in lname:
            buckets["ffn"] += count

        elif "layernorm" in lname or "layer_norm" in lname:
            buckets["norms"] += count

        elif "pooler" in lname:
            buckets["pooler"] += count

        else:
            buckets["other"] += count

    buckets["total"] = sum(buckets.values())

    return buckets


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt)
        print(audit(ckpt))
        print()