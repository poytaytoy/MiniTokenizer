use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::Deserialize;

pub struct TokenDecoder {
    vocab: HashMap<u32, String>,
}

impl TokenDecoder {
    pub fn new(tokenizer_path: &str) -> Result<Self> {
        let tokenizer_json_str = std::fs::read_to_string(tokenizer_path)?;

        #[derive(Deserialize, Debug)]
        struct BpeTokenizer {
            pub vocab: HashMap<String, usize>,
        }

        let bpe_tokenizer: BpeTokenizer = serde_json::from_str(&tokenizer_json_str)?;

        let mut vocab = HashMap::new();
        for (token, &id) in bpe_tokenizer.vocab.iter() {
            vocab.insert(id as u32, token.clone());
        }

        Ok(TokenDecoder { vocab })
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut decoded = String::new();
        for &token_id in tokens {
            let token_str = self.vocab.get(&token_id)
                .ok_or_else(|| anyhow!("Token ID {} not found in vocabulary", token_id))?;
            decoded.push_str(token_str);
        }
        Ok(decoded)
    }
}