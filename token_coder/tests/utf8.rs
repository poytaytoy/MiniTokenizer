use token_coder::{decode::TokenDecoder, encode::TokenEncoder, wordarena::WordArena};

const TOKENIZER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/utf8_tokenizer.json"
);

#[test]
fn unicode_round_trips_through_both_merge_paths_and_cache() {
    let mut encoder = TokenEncoder::new(TOKENIZER).unwrap();
    let decoder = TokenDecoder::new(TOKENIZER).unwrap();
    for text in [
        String::new(),
        "hello café 世界 🦀🙂 e\u{301}\t\nПривет مرحبا नमस्ते".to_owned(),
        "café".repeat(100),
        "世界".repeat(100),
        "🦀🙂".repeat(100),
    ] {
        for _ in 0..2 {
            let tokens = encoder.encode(&text);
            assert_eq!(decoder.decode(&tokens).unwrap(), text);
        }
    }
    // Both ASCII and multibyte characters participate in BPE merges.
    for text in ["café", "世界", "🦀🙂"] {
        assert_eq!(encoder.encode(text).len(), 1);
    }
}

#[test]
fn arena_merges_at_character_boundaries() {
    let mut arena = WordArena::new("é中🦀a".as_bytes(), 0);
    assert_eq!(arena.len(), 4);
    for (index, character) in ["é", "中", "🦀", "a"].iter().enumerate() {
        assert_eq!(arena.get_word_splice(index), character.as_bytes());
    }
    arena.merge_next(arena[0]);
    assert_eq!(arena.get_word_splice(0), "é中".as_bytes());
    assert!(arena.cmp_from_u8(arena[0], "é中".as_bytes()));
    assert_eq!(arena.get_back(arena[2]), Some(arena[0]));
}

#[test]
fn decoder_rejects_unknown_ids() {
    let decoder = TokenDecoder::new(TOKENIZER).unwrap();
    assert!(decoder.decode(&[u32::MAX]).is_err());
}
