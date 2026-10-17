use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;

use crate::REGEX_GPT_2;
use crate::splice::*;
use crate::wordarena::*;
use anyhow::Result;
use regex::Regex;
use serde::Deserialize;

pub struct TokenEncoder {
    vocab: HashMap<Splice, usize>,
    merges: HashMap<(Splice, Splice), usize>,
    splice_db: SpliceDatabase,
    cache_db: HashMap<Vec<u8>, Vec<u32>>,
    regex: Regex,
}

impl TokenEncoder {
    pub fn new(tokenizer_path: &str) -> Result<TokenEncoder> {
        let tokenizer_json_str = std::fs::read_to_string(tokenizer_path)?;

        #[derive(Deserialize, Debug)]
        struct BpeTokenizer {
            pub vocab: HashMap<String, usize>,
            pub merges: Vec<[String; 2]>,
        }

        let bpe_tokenizer: BpeTokenizer = serde_json::from_str(&tokenizer_json_str)?;

        let mut splice_db = SpliceDatabase::new();
        let mut vocab: HashMap<Splice, usize> = HashMap::new();
        let mut merges: HashMap<(Splice, Splice), usize> = HashMap::new();

        for (token, value) in bpe_tokenizer.vocab.iter() {
            let token: Vec<u8> = token.clone().into_bytes();
            vocab.insert(splice_db.insert(&token), value.clone());
        }

        for (rank, [token1, token2]) in bpe_tokenizer.merges.iter().enumerate() {
            let token1 = splice_db.get(&token1.clone().as_bytes()).unwrap();
            let token2 = splice_db.get(&token2.clone().as_bytes()).unwrap();
            merges.insert((token1, token2), rank);
        }

        let regex = Regex::new(REGEX_GPT_2)?;

        Ok(TokenEncoder {
            vocab: vocab,
            merges: merges,
            splice_db: splice_db,
            cache_db: HashMap::new(),
            regex: regex,
        })
    }

    fn convert_final_arena(&self, arena: WordArena) -> Vec<u32> {
        let mut tokenized_word = Vec::new();
        let mut curr_token = arena[0];

        loop {
            let tokenized = self
                .vocab
                .get(arena.get_word_splice(curr_token.start))
                .unwrap();
            tokenized_word.push(tokenized.clone() as u32);

            curr_token = match arena.get_next(curr_token) {
                Some(token) => token,
                None => {
                    break;
                }
            }
        }

        return tokenized_word;
    }

    fn get_pair_ranking(&self, pair: (Token, Token), arena: &WordArena) -> usize {
        let (token, next) = pair;

        let splice1 = self
            .splice_db
            .get(arena.get_word_splice(token.start))
            .unwrap();
        let splice2 = self
            .splice_db
            .get(arena.get_word_splice(next.start))
            .unwrap();

        return match self.merges.get(&(splice1, splice2)) {
            Some(rank) => rank.clone(),
            None => usize::MAX,
        };
    }

    fn naive_merge(&mut self, mut arena: WordArena) -> Vec<u32> {
        let mut best_token = arena[0];
        let mut best_rank = usize::MAX;

        for index in 0..arena.len() {
            let token = arena[index];
            if !token.alive {
                continue;
            }
            let next = match arena.get_next(token) {
                Some(next_token) => next_token,
                None => continue,
            };

            let rank = self.get_pair_ranking((token, next), &arena);

            if rank < best_rank {
                best_rank = rank.clone();
                best_token = token;
            }
        }

        if best_rank == usize::MAX {
            return self.convert_final_arena(arena);
        }

        arena.merge_next(best_token);
        return self.naive_merge(arena);
    }

    fn optimized_merge(&mut self, mut arena: WordArena) -> Vec<u32> {
        let mut order_heap: BinaryHeap<Reverse<(usize, (Token, Token))>> = BinaryHeap::new();

        // First load of the order_heap
        for index in 0..arena.len() {
            let token = arena[index];
            if !token.alive {
                continue;
            }
            let next = match arena.get_next(token) {
                Some(next_token) => next_token,
                None => continue,
            };

            let splice1 = self
                .splice_db
                .get(arena.get_word_splice(token.start))
                .unwrap();
            let splice2 = self
                .splice_db
                .get(arena.get_word_splice(next.start))
                .unwrap();

            if let Some(rank) = self.merges.get(&(splice1, splice2)) {
                order_heap.push(Reverse((*rank, (token, next))));
            }
        }

        let is_valid = |pairs: (Token, Token), arena: &WordArena| -> bool {
            let (start, next) = pairs;
            return start == arena[start.start] && next == arena[next.start];
        };

        let mut best_heap_item = order_heap.pop();

        loop {
            if best_heap_item.is_none() {
                return self.convert_final_arena(arena);
            }

            let Reverse((_, (start, next))) = best_heap_item.unwrap();

            if !is_valid((start, next), &arena) {
                best_heap_item = order_heap.pop();
                continue;
            }

            arena.merge_next(start);

            let curr_start = arena[start.start];
            let curr_back = arena.get_back(curr_start);
            let curr_next = arena.get_next(curr_start);

            if curr_back.is_some() {
                let curr_back = curr_back.unwrap();
                let pair = (curr_back, curr_start);
                let rank = self.get_pair_ranking(pair.clone(), &arena);

                if rank != usize::MAX {
                    order_heap.push(Reverse((rank, pair)))
                }
            }

            if curr_next.is_some() {
                let curr_next = curr_next.unwrap();
                let pair = (curr_start, curr_next);
                let rank = self.get_pair_ranking(pair.clone(), &arena);

                if rank != usize::MAX {
                    order_heap.push(Reverse((rank, pair)))
                }
            }

            best_heap_item = order_heap.pop();
        }
    }

    fn merge_token(&mut self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        if let Some(cached_result) = self.cache_db.get(bytes) {
            return cached_result.clone();
        }

        if bytes.len() < 100 {
            let result = self.naive_merge(WordArena::new(bytes, 0));
            self.cache_db.insert(bytes.to_vec(), result.clone());
            return result;
        } else {
            return self.optimized_merge(WordArena::new(bytes, 0));
        }
    }

    pub fn encode(&mut self, message: &str) -> Vec<u32> {
        let mut output = Vec::new();

        let tokens: Vec<&str> = self.regex.find_iter(message).map(|m| m.as_str()).collect();

        for token in tokens {
            output.append(&mut self.merge_token(token.as_bytes()));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::TokenDecoder;

    #[test]
    fn test_encode_decode_naive_merge() {
        let mut encoder = TokenEncoder::new("test_tokenizer.json").unwrap();
        let decoder = TokenDecoder::new("test_tokenizer.json").unwrap();

        let original = "hello world";
        let encoded = encoder.encode(original);
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_optimized_merge() {
        let mut encoder = TokenEncoder::new("test_tokenizer.json").unwrap();
        let decoder = TokenDecoder::new("test_tokenizer.json").unwrap();

        let original = "ab".repeat(120);
        let encoded = encoder.encode(&original);
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_encode_decode_mixed() {
        let mut encoder = TokenEncoder::new("test_tokenizer.json").unwrap();
        let decoder = TokenDecoder::new("test_tokenizer.json").unwrap();

        let original = format!(
            "short message followed by a very long word: {}",
            "a".repeat(150)
        );
        let encoded = encoder.encode(&original);
        let decoded = decoder.decode(&encoded).unwrap();

        assert_eq!(decoded, original);
    }
}
