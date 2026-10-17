import json
import unittest

from tokenizer import MiniTokenizer
from python.dataset import load_text, split_text


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = "hello café 世界 🦀🙂 e\u0301\t\nПривет مرحبا नमस्ते "
        cls.tokenizer = MiniTokenizer.train(cls.text * 5, 200)

    def test_utf8_and_both_merge_paths(self):
        for text in [self.text, "café" * 100, "世界" * 100, "🦀🙂" * 100, ""]:
            with self.subTest(text=text[:20]):
                for _ in range(2):  # Exercise the cache too.
                    ids = self.tokenizer.encode(text)
                    self.assertIsInstance(ids, list)
                    self.assertTrue(all(isinstance(i, int) for i in ids))
                    self.assertEqual(self.tokenizer.decode(ids), text)

    def test_export_reload_string(self):
        model = self.tokenizer.to_json()
        self.assertIn("merges", json.loads(model))
        loaded = MiniTokenizer.load(model)
        self.assertEqual(loaded.encode(self.text), self.tokenizer.encode(self.text))
        self.assertEqual(loaded.decode(loaded.encode(self.text)), self.text)

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode([2**32 - 1])
        with self.assertRaises(ValueError):
            self.tokenizer.encode("🚀")
        for model in ["not json", '{"vocab": {}, "merges": [["a", "b"]]}']:
            with self.subTest(model=model), self.assertRaises(RuntimeError):
                MiniTokenizer.load(model)

    def test_empty_training(self):
        tokenizer = MiniTokenizer.train("", 1000)
        self.assertEqual(tokenizer.decode(tokenizer.encode("ASCII\n")), "ASCII\n")
        self.assertEqual(len(json.loads(tokenizer.to_json())["vocab"]), 128)

    def test_vocabulary_target_and_exhaustion(self):
        model = json.loads(MiniTokenizer.train("aaaaaa", 130).to_json())
        self.assertEqual(len(model["vocab"]), 130)
        self.assertTrue(model["merges"])
        self.assertEqual(len(json.loads(MiniTokenizer.train("", 1).to_json())["vocab"]), 128)


class DatasetTests(unittest.TestCase):
    def test_train_and_round_trip_evaluation_text(self):
        training, evaluation = split_text(load_text())
        tokenizer = MiniTokenizer.train(training, 1000)
        loaded = MiniTokenizer.load(tokenizer.to_json())
        ids = loaded.encode(evaluation)
        self.assertEqual(loaded.decode(ids), evaluation)
        self.assertEqual(tokenizer.encode(evaluation), ids)
        self.assertLess(len(ids), len(evaluation))


if __name__ == "__main__":
    unittest.main()
