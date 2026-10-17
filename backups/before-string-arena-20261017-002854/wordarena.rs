use crate::character::extract_char;
use std::ops::Index;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Token {
    pub back: usize, // usize::MIN for the there is no back
    pub start: usize,
    pub next: usize, // usize::MAX to indicate there is no next,
    pub alive: bool,
}

#[derive(Debug)]
pub struct WordArena {
    byte_offsets: Vec<usize>,
    pub word: Vec<u8>, // Make it a vec with a fixed capacity
    pub token_arena: Vec<Token>,
    pub count: usize,
    pub id: usize,
}

impl WordArena {
    pub fn new(word: &[u8], id: usize) -> WordArena {
        let characters = extract_char(word).expect("WordArena requires valid UTF-8");
        let mut byte_offsets = Vec::with_capacity(characters.len());
        let mut offset = 0;
        for character in &characters {
            byte_offsets.push(offset);
            offset += character.character.len();
        }
        let word = word.to_vec();
        let mut token_arena = Vec::with_capacity(characters.len());

        for i in 0..characters.len() {
            let back = if i == 0 { usize::MAX } else { i - 1 };
            let next = if i + 1 >= characters.len() {
                usize::MAX
            } else {
                i + 1
            };

            token_arena.push(Token {
                back: back,
                start: i,
                next: next,
                alive: true,
            });
        }

        WordArena {
            byte_offsets,
            word: word,
            token_arena: token_arena,
            count: 0,
            id: id,
        }
    }

    // There shouldn't be a need to decrement since you can't get rid of a word.
    pub fn increment_count(&mut self) {
        self.count += 1;
    }

    pub fn merge_next(&mut self, token: Token) {
        let mut next = match self.get_next(token.clone()) {
            Some(token) => token,
            None => {
                return;
            }
        };

        let mut start = token;

        start.next = next.next;
        next.alive = false;

        if let Some(mut next_next) = self.get_next(next) {
            next_next.back = start.start;
            self.token_arena[next_next.start] = next_next;
        }

        self.token_arena[start.start] = start;
        self.token_arena[next.start] = next;
    }

    pub fn get_next(&self, token: Token) -> Option<Token> {
        if token.next == usize::MAX {
            return None;
        }
        Some(self.token_arena[token.next])
    }

    pub fn get_back(&self, token: Token) -> Option<Token> {
        if token.back == usize::MAX {
            return None;
        }
        Some(self.token_arena[token.back])
    }

    fn _next(&self, next: usize) -> usize {
        if next == usize::MAX {
            return self.word.len();
        }
        self.byte_offsets[next]
    }

    pub fn cmp_from_u8(&self, token: Token, cmp: &[u8]) -> bool {
        let next = self._next(token.next);
        let token = &self.word[self.byte_offsets[token.start]..next];
        cmp == token
    }

    pub fn get_word_splice(&self, index: usize) -> &[u8] {
        let token = &self.token_arena[index];
        let end = self._next(token.next);
        &self.word[self.byte_offsets[token.start]..end]
    }

    pub fn len(&self) -> usize {
        self.token_arena.len()
    }
}

impl Index<usize> for WordArena {
    type Output = Token;
    fn index(&self, index: usize) -> &Self::Output {
        &self.token_arena[index]
    }
}

#[cfg(test)]
mod test_word_arena {
    use super::*;

    #[test]
    fn test_word_arena_new() {
        let word_arena = WordArena::new(&[1, 2, 3], 0);
        assert_eq!(word_arena.word, &[1, 2, 3]);
        assert_eq!(word_arena.count, 0);
        assert_eq!(word_arena.id, 0);
        assert_eq!(word_arena.token_arena.len(), 3);
    }

    #[test]
    fn test_merge_next_alive_checker() {
        let mut word_arena = WordArena::new(&[1, 2, 3], 0);
        let first = word_arena[0];

        word_arena.merge_next(first);
        assert_eq!(word_arena[0].next, 2);
        assert!(!word_arena[1].alive);

        assert_eq!(word_arena[2].back, 0);
        assert!(word_arena[2].alive);

        assert_eq!(word_arena.get_word_splice(0), &[1, 2]);
    }
}
