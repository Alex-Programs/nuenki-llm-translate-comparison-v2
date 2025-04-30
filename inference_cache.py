import json
from typing import Optional
from typedefinitions import TranslatableLanguage

_CACHE_FILE = "cache.json"


def write_cache(
    entry_id: str,
    source_lang: TranslatableLanguage,
    target_lang: TranslatableLanguage,
    text: str,
    temperature: float,
    response: str,
) -> None:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    cache[entry_id] = {
        "source_lang": source_lang.value,
        "target_lang": target_lang.value,
        "text": text,
        "temperature": temperature,
        "response": response,
    }

    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=4)


def pull_cache(
    entry_id: str,
    source_lang: TranslatableLanguage,
    target_lang: TranslatableLanguage,
    text: str,
    temperature: float,
) -> Optional[str]:
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    entry = cache.get(entry_id)
    if not entry:
        return None

    if (
        entry.get("source_lang") == source_lang.value
        and entry.get("target_lang") == target_lang.value
        and entry.get("text") == text
        and entry.get("temperature") == temperature
    ):
        return entry.get("response")
    return None
