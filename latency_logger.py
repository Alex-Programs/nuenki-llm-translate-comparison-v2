import json
import time


def log_latency(record_id: str, duration: float) -> None:
    entry = {
        "id": record_id,
        "duration": duration,
        "logged_at": time.time(),
    }
    with open("latency.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
