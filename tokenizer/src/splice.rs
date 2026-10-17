use std::borrow::Borrow;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub struct Splice {
    splice: Rc<Vec<u8>>,
}

impl Splice {
    pub fn new(splice: &[u8]) -> Splice {
        Splice {
            splice: Rc::new(splice.to_vec()),
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        self.splice.as_slice()
    }
}

// Needed for the Hashmap
impl Borrow<[u8]> for Splice {
    fn borrow(&self) -> &[u8] {
        self.splice.as_slice()
    }
}

pub struct SpliceDatabase {
    pub splice_db: HashMap<Splice, Splice>,
}

impl SpliceDatabase {
    pub fn new() -> SpliceDatabase {
        SpliceDatabase {
            splice_db: HashMap::new(),
        }
    }

    // Returns the canonical Splice for these bytes, interning it if it hasn't been seen before
    pub fn insert(&mut self, splice: &[u8]) -> Splice {
        if let Some(existing) = self.splice_db.get(splice) {
            return existing.clone();
        }

        let splice_ref = Splice::new(splice);
        self.splice_db
            .insert(splice_ref.clone(), splice_ref.clone());
        splice_ref
    }

    pub fn get(&self, splice: &[u8]) -> Option<Splice> {
        self.splice_db.get(splice).cloned()
    }
}
