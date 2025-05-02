import json
import time


def log_latency(
    model_id: str, sentence_cat: str, sentence: str, duration: float
) -> None:
    entry = {
        "model_id": model_id,
        "sentence_cat": sentence_cat,
        "sentence": sentence,
        "duration": duration,
        "logged_at": time.time(),
    }
    with open("latency.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
