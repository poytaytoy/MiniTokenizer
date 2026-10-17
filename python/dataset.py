"""Download and cache the complete Project Gutenberg Alice text (stdlib only)."""
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
CACHE = Path(__file__).resolve().parent / "data" / "alice.txt"


def load_text():
    if not CACHE.exists():
        request = Request(URL, headers={"User-Agent": "tokenizer-benchmark/0.1"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        content.decode("utf-8-sig")  # Validate before caching.
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(content)
    return CACHE.read_text(encoding="utf-8-sig")


def split_text(text):
    """80% training, 20% evaluation; include the full character alphabet in training."""
    split = int(len(text) * 0.8)
    training, evaluation = text[:split], text[split:]
    # Expose characters, but no evaluation words or adjacent pairs, to the model.
    missing = sorted(set(evaluation) - set(training))
    training += "\n" + "\n".join(missing)
    return training, evaluation


def describe(text):
    return {"source": URL, "bytes": len(text.encode("utf-8")),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


if __name__ == "__main__":
    import json
    print(json.dumps(describe(load_text()), indent=2))
