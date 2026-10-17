use anyhow::Result;
use regex::Regex;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::REGEX_GPT_2;
use crate::splice::*;
use crate::wordarena::*;

struct TokenIdMap {
    tokenid_map: HashMap<Splice, usize>,
    rankings: Vec<(Splice, Splice)>,
}

impl TokenIdMap {
    fn new(splice_db: &mut SpliceDatabase) -> TokenIdMap {
        let mut tokenid_map = HashMap::new();

        // Insert all standard ASCII bytes
        for i in 0..=127u8 {
            let splice = splice_db.insert(&[i]);
            tokenid_map.insert(splice, i as usize);
        }

        TokenIdMap {
            tokenid_map: tokenid_map,
            rankings: Vec::new(),
        }
    }

    fn insert(&mut self, pair: (Splice, Splice), merged: Splice) {
        let id = self.tokenid_map.len();
        self.tokenid_map.entry(merged).or_insert(id);
        self.rankings.push(pair);
    }

    fn len(&self) -> usize {
        self.tokenid_map.len()
    }

    fn to_json(self) -> Result<String> {
        #[derive(Serialize)]
        struct BpeModel {
            vocab: HashMap<String, usize>,
            merges: Vec<(String, String)>,
        }

        // All tokens are formed by joining valid UTF-8 characters.
        let merges: Vec<(String, String)> = self
            .rankings
            .iter()
            .map(|(start, next)| {
                Ok((
                    std::str::from_utf8(start.as_slice())?.to_owned(),
                    std::str::from_utf8(next.as_slice())?.to_owned(),
                ))
            })
            .collect::<Result<_>>()?;

        let vocab: HashMap<String, usize> = self
            .tokenid_map
            .iter()
            .map(|(splice, id)| Ok((std::str::from_utf8(splice.as_slice())?.to_owned(), *id)))
            .collect::<Result<_>>()?;

        let tokenizer = BpeModel {
            vocab: vocab,
            merges: merges,
        };
        Ok(serde_json::to_string_pretty(&tokenizer)?)
    }
}

struct WordMap {
    word_map: Vec<WordArena>,
    id_map: HashMap<Splice, usize>,
}

impl WordMap {
    fn new() -> WordMap {
        WordMap {
            word_map: Vec::new(),
            id_map: HashMap::new(),
        }
    }

    fn add_word(&mut self, word: &[u8], splice_db: &mut SpliceDatabase) {
        let option = self.id_map.get(word);

        match option {
            Some(index) => {
                let arena = &mut self.word_map[index.clone()];
                arena.increment_count();
            }
            None => {
                let index = self.word_map.len();
                let mut arena = WordArena::new(word, index);
                arena.increment_count();

                self.word_map.push(arena);
                self.id_map.insert(splice_db.insert(word), index);
            }
        }
    }

    fn get_arena_by_index_mutable(&mut self, index: usize) -> &mut WordArena {
        &mut self.word_map[index]
    }
}

// Hashmap stores each token's location
struct LocationMap {
    location_map: HashMap<Splice, Vec<Location>>,
}

impl LocationMap {
    fn new() -> LocationMap {
        LocationMap {
            location_map: HashMap::new(),
        }
    }

    fn populate_from_arena(&mut self, arena: &WordArena, splice_db: &mut SpliceDatabase) {
        for token in &arena.token_arena {
            if !token.alive {
                continue;
            }

            let word = arena.get_word_splice(token.start);

            let entry_reference;

            if self.location_map.contains_key(word) {
                entry_reference = self.location_map.get_mut(word).unwrap();
            } else {
                self.location_map.insert(splice_db.insert(word), Vec::new());
                entry_reference = self.location_map.get_mut(word).unwrap();
            }

            entry_reference.push(Location {
                word_index: arena.id,
                arena_token: token.clone(),
                active: true,
            })
        }
    }

    fn get_mut(&mut self, token: &Splice) -> &mut Vec<Location> {
        self.location_map.get_mut(token).unwrap()
    }

    fn insert(&mut self, token: Splice, locations: Vec<Location>) {
        self.location_map
            .entry(token)
            .or_default()
            .extend(locations);
    }
}

#[derive(Copy, Clone, Debug)]
struct Location {
    word_index: usize,
    arena_token: Token,
    active: bool,
}

// Hashmap storing the count of each index
struct PairCountMap {
    pair_count_map: HashMap<(Splice, Splice), usize>,
}

impl PairCountMap {
    fn new() -> PairCountMap {
        PairCountMap {
            pair_count_map: HashMap::new(),
        }
    }

    fn get_mut_count(&mut self, pair: (Splice, Splice)) -> &mut usize {
        self.pair_count_map.entry(pair).or_insert(0)
    }

    fn get_count(&self, pair: &(Splice, Splice)) -> usize {
        self.pair_count_map.get(pair).unwrap().clone()
    }
}

struct BestPairQueue {
    priority_queue: BinaryHeap<Pair>,
}

impl BestPairQueue {
    fn new() -> BestPairQueue {
        BestPairQueue {
            priority_queue: BinaryHeap::new(),
        }
    }

    fn insert(&mut self, pair: (Splice, Splice), count: usize) {
        self.priority_queue.push(Pair {
            pair: pair,
            count: count,
        });
    }

    fn pair_valid(&self, pair: &Pair, count_map: &PairCountMap) -> bool {
        pair.count == count_map.get_count(&pair.pair)
    }

    fn pop(&mut self, count_map: &PairCountMap) -> Option<Pair> {
        loop {
            if let Some(pair) = self.priority_queue.pop() {
                if self.pair_valid(&pair, count_map) {
                    return Some(pair);
                }
            } else {
                return None;
            }
        }
    }

    fn push(&mut self, pair: Pair) {
        self.priority_queue.push(pair);
    }

    fn update(&mut self, pair: (Splice, Splice), count_map: &PairCountMap) {
        let pair_read = &pair;
        let count = count_map.get_count(pair_read);
        self.push(Pair { pair, count })
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Pair {
    pair: (Splice, Splice),
    count: usize,
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.0.as_slice().cmp(other.pair.0.as_slice()))
            .then_with(|| self.pair.1.as_slice().cmp(other.pair.1.as_slice()))
    }
}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BPETrainer {
    token_id_map: TokenIdMap,
    word_id_map: WordMap,
    token_pair_to_count: PairCountMap,
    token_pair_pqueue: BestPairQueue,
    token_to_location: LocationMap,
    splice_database: SpliceDatabase,
}

impl BPETrainer {
    pub fn new(text: &str) -> Result<Self> {
        let mut splice_database = SpliceDatabase::new();
        let token_id_map = TokenIdMap::new(&mut splice_database);

        let mut trainer = Self {
            token_id_map,
            word_id_map: WordMap::new(),
            token_pair_to_count: PairCountMap::new(),
            token_pair_pqueue: BestPairQueue::new(),
            token_to_location: LocationMap::new(),
            splice_database,
        };

        trainer.build_vocab(text)?;
        trainer.setup_maps()?;

        Ok(trainer)
    }

    fn build_vocab(&mut self, text: &str) -> Result<()> {
        let regex = Regex::new(REGEX_GPT_2)?;
        for character in text.chars() {
            let mut buffer = [0; 4];
            let bytes = character.encode_utf8(&mut buffer).as_bytes();
            let id = self.token_id_map.tokenid_map.len();
            self.token_id_map
                .tokenid_map
                .entry(self.splice_database.insert(bytes))
                .or_insert(id);
        }
        for token in regex.find_iter(text) {
            self.word_id_map
                .add_word(token.as_str().as_bytes(), &mut self.splice_database);
        }
        Ok(())
    }

    fn setup_maps(&mut self) -> Result<()> {
        // Loops through each word and logs all the tokens in the location
        // And count all pairs
        for arena in &self.word_id_map.word_map {
            self.token_to_location
                .populate_from_arena(arena, &mut self.splice_database);

            let mut current_index = 0;

            loop {
                let current_token = arena.token_arena[current_index];
                let next_index = current_token.next;

                if next_index == usize::MAX {
                    break;
                }

                let start_splice = self
                    .splice_database
                    .insert(arena.get_word_splice(current_index));
                let next_splice = self
                    .splice_database
                    .insert(arena.get_word_splice(next_index));

                let count = self
                    .token_pair_to_count
                    .get_mut_count((start_splice, next_splice));

                *count += arena.count;

                current_index = next_index;
            }
        }

        // push the initial set of pairs
        for (pair, count) in self.token_pair_to_count.pair_count_map.iter() {
            self.token_pair_pqueue.insert(pair.clone(), count.clone());
        }

        Ok(())
    }

    fn update_pair_count(&mut self, update_count: usize, increment: bool, pair: (Splice, Splice)) {
        let count = self.token_pair_to_count.get_mut_count(pair.clone());

        if increment {
            *count += update_count;
        } else {
            *count -= update_count;
        }

        self.token_pair_pqueue
            .update(pair, &self.token_pair_to_count);
    }

    fn update_maps(&mut self) -> Result<bool> {
        let best_pair = match self.token_pair_pqueue.pop(&self.token_pair_to_count) {
            Some(pair) => pair,
            None => {
                return Ok(false);
            }
        };

        if best_pair.count == 0 {
            return Ok(false);
        }
        let (start, next) = best_pair.pair;
        let merged_bytes = [start.as_slice(), next.as_slice()].concat();
        let merged_token = self.splice_database.insert(&merged_bytes);
        let mut merged_token_location: Vec<Location> = Vec::new();

        // Update the WordMap
        let mut locations = Vec::new();

        // We swap the super big location mut into the empty locations. This way we can BEAT the borrow checker
        // without a HUGE clone
        std::mem::swap(&mut locations, self.token_to_location.get_mut(&start));

        for loc in locations.iter_mut() {
            if !loc.active {
                continue;
            }

            let arena = self.word_id_map.get_arena_by_index_mutable(loc.word_index);

            let current_token = arena[loc.arena_token.start];

            if !current_token.alive
                || arena.get_word_splice(loc.arena_token.start) != start.as_slice()
            {
                loc.active = false;
                continue;
            }

            let neighbor = match arena.get_next(current_token) {
                Some(neighbor_token) => neighbor_token,
                None => {
                    continue;
                }
            };

            if !arena.cmp_from_u8(neighbor, next.as_slice()) {
                continue;
            }

            arena.merge_next(current_token);

            // The count from (back, start) and (next, next_next) needs to be updated and we need to increment the new one
            // back/next_next may not exist if this merge happened at the start/end of the word
            let back_splice = if current_token.back != usize::MAX {
                Some(
                    self.splice_database
                        .insert(arena.get_word_splice(current_token.back)),
                )
            } else {
                None
            };

            let next_next_splice = if neighbor.next != usize::MAX {
                Some(
                    self.splice_database
                        .insert(arena.get_word_splice(neighbor.next)),
                )
            } else {
                None
            };

            let arena_count = arena.count.clone();

            // Add the new entry to the new hashmap
            merged_token_location.push(Location {
                word_index: loc.word_index,
                // Get the updated entry
                arena_token: arena[loc.arena_token.start],
                active: true,
            });

            // Weird ordering because we need to take all the mutable references out before we can do this

            if let Some(back_splice) = &back_splice {
                // Decrement unused
                self.update_pair_count(arena_count, false, (back_splice.clone(), start.clone()));

                // Increment used
                self.update_pair_count(
                    arena_count,
                    true,
                    (back_splice.clone(), merged_token.clone()),
                );
            }

            if let Some(next_next_splice) = &next_next_splice {
                // Decrement unused
                self.update_pair_count(
                    arena_count,
                    false,
                    (next.clone(), next_next_splice.clone()),
                );

                // Increment used
                self.update_pair_count(
                    arena_count,
                    true,
                    (merged_token.clone(), next_next_splice.clone()),
                );
            }
        }

        // Swap back the memory in place.
        std::mem::swap(self.token_to_location.get_mut(&start), &mut locations);

        // Register where the newly merged token now lives
        self.token_to_location
            .insert(merged_token.clone(), merged_token_location);

        // Remove the count for the unmerged pair
        let pair = (start, next);
        let pair_count = self.token_pair_to_count.get_count(&pair);

        self.update_pair_count(pair_count, false, pair.clone());

        // Insert the new token
        self.token_id_map.insert(pair, merged_token);

        Ok(true)
    }

    pub fn train(&mut self, vocab_size: usize) -> Result<()> {
        while self.token_id_map.len() < vocab_size && self.update_maps()? {}
        Ok(())
    }

    pub fn to_json(self) -> Result<String> {
        self.token_id_map.to_json()
    }
}

pub fn train_bpe_tokenizer(text: &str, vocab_size: usize) -> Result<String> {
    let mut training = BPETrainer::new(text)?;
    training.train(vocab_size)?;
    training.to_json()
}
