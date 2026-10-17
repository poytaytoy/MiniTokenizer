"""Run from the repository root: python -m python.benchmark"""
from time import perf_counter

from tokenizer import MiniTokenizer
from python.dataset import load_text, split_text


if __name__ == "__main__":
    training, text = split_text(load_text())
    size = len(text.encode("utf-8"))

    start = perf_counter()
    tokenizer = MiniTokenizer.train(training, 1000)
    print(f"Training: {(perf_counter() - start) * 1000:.2f} ms")
    print(f"Training bytes: {len(training.encode('utf-8'))}; evaluation bytes: {size}")

    start = perf_counter()
    ids = tokenizer.encode(text)
    elapsed = perf_counter() - start
    print(f"First encode: {elapsed * 1000:.2f} ms ({size / elapsed / 1e6:.2f} MB/s)")
    assert tokenizer.decode(ids) == text

    start = perf_counter()
    for _ in range(100):
        tokenizer.encode(text)
    elapsed = (perf_counter() - start) / 100
    print(f"Cached encode: {elapsed * 1000:.2f} ms ({size / elapsed / 1e6:.2f} MB/s)")

    start = perf_counter()
    for _ in range(100):
        tokenizer.decode(ids)
    elapsed = (perf_counter() - start) / 100
    print(f"Decode: {elapsed * 1000:.2f} ms ({size / elapsed / 1e6:.2f} MB/s)")
    print(f"Round trip passed; {len(ids)} tokens")
