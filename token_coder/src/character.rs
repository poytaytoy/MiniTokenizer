use anyhow::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Character {
    pub character: Vec<u8>,
}

impl Character {
    pub fn new(bytes: Vec<u8>) -> Character {
        Character { character: bytes }
    }
}

/// Split valid UTF-8 into Unicode scalar values, rejecting malformed input.
pub fn extract_char(bytes: &[u8]) -> Result<Vec<Character>> {
    let text = std::str::from_utf8(bytes)?;
    Ok(text
        .chars()
        .map(|ch| {
            let mut buffer = [0; 4];
            Character::new(ch.encode_utf8(&mut buffer).as_bytes().to_vec())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_all_utf8_widths() {
        let characters = extract_char("Aé中🦀".as_bytes()).unwrap();
        let bytes: Vec<_> = characters
            .iter()
            .map(|ch| ch.character.as_slice())
            .collect();
        assert_eq!(
            bytes,
            vec![
                "A".as_bytes(),
                "é".as_bytes(),
                "中".as_bytes(),
                "🦀".as_bytes()
            ]
        );
        assert!(extract_char(b"").unwrap().is_empty());
    }

    #[test]
    fn rejects_invalid_utf8() {
        for bytes in [
            &[0x80][..],
            &[0xc0, 0xaf],
            &[0xe2, 0x82],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
        ] {
            assert!(extract_char(bytes).is_err());
        }
    }
}
