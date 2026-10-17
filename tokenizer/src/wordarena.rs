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
    pub word: String,
    pub token_arena: Vec<Token>,
    pub count: usize,
    pub id: usize,
}

impl WordArena {
    pub fn new(word: &[u8], id: usize) -> WordArena {
        let word = std::str::from_utf8(word)
            .expect("WordArena requires valid UTF-8")
            .to_owned();
        let byte_offsets: Vec<usize> = word.char_indices().map(|(offset, _)| offset).collect();
        let character_count = byte_offsets.len();
        let mut token_arena = Vec::with_capacity(character_count);

        for i in 0..character_count {
            let back = if i == 0 { usize::MAX } else { i - 1 };
            let next = if i + 1 >= character_count {
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
        let token = &self.word.as_bytes()[self.byte_offsets[token.start]..next];
        cmp == token
    }

    pub fn get_word_splice(&self, index: usize) -> &[u8] {
        let token = &self.token_arena[index];
        let end = self._next(token.next);
        &self.word.as_bytes()[self.byte_offsets[token.start]..end]
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
