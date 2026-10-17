# MiniTokenizer

A Rust BPE tokenizer with a Python API and UTF-8 support.

```sh
python -m pip install .
```

```python
from tokenizer import MiniTokenizer

tokenizer = MiniTokenizer.train("hello world", vocab_size=1000)
ids = tokenizer.encode("hello world")
text = tokenizer.decode(ids)

# Export and reload a model using JSON strings.
model = tokenizer.to_json()
tokenizer = MiniTokenizer.load(model)
```

Training learns frequent character merges. Encoding turns text into token IDs;
decoding turns IDs back into text. The model must contain the characters you
encode. Keep each instance on the thread that created it.

## Benchmark

Python API, release build, 1,000-token vocabulary. Alice text: 137 KB for training
and 34 KB for evaluation. Evaluation-only characters are included in the training
alphabet.

| Operation | Time | Throughput |
| --- | ---: | ---: |
| Training | 59.79 ms | — |
| First encode | 9.41 ms | 3.56 MB/s |
| Cached encode | 1.68 ms | 19.96 MB/s |
| Decode | 0.75 ms | 44.48 MB/s |

Training and first encode were measured once; cached encode and decode are
averages over 100 calls. Download time is excluded. Round-trip verification passed.

```sh
python -m python.benchmark
python -m unittest discover -s python/tests
```
