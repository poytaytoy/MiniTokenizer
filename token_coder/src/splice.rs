use std::rc::Rc; 
use std::borrow::Borrow;
use std::collections::HashMap;

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
pub struct Splice {
    splice: Rc<Vec<u8>>
}

impl Splice {
    pub fn new(splice: &[u8]) -> Splice {
        Splice { splice: Rc::new(splice.to_vec()) }
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

#[cfg(test)]
mod test_splice {
    use super::*;

    #[test]
    fn test_splice_new() {
        let splice = Splice::new(&[1, 2, 3]);
        assert_eq!(splice.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_splice_borrow() {
        let splice = Splice::new(&[1, 2, 3]);
        let borrowed: &[u8] = splice.borrow();
        assert_eq!(borrowed, &[1, 2, 3]);
    }
}

pub struct SpliceDatabase{
    pub splice_db: HashMap<Splice, Splice>
}

impl SpliceDatabase{
    pub fn new() -> SpliceDatabase {
        SpliceDatabase { splice_db: HashMap::new() }
    }

    // Returns the canonical Splice for these bytes, interning it if it hasn't been seen before
    pub fn insert(&mut self, splice: &[u8]) -> Splice {
        if let Some(existing) = self.splice_db.get(splice) {
            return existing.clone();
        }

        let splice_ref = Splice::new(splice);
        self.splice_db.insert(splice_ref.clone(), splice_ref.clone());
        splice_ref
    }

    pub fn get(&self, splice: &[u8]) -> Option<Splice>{
        self.splice_db.get(splice).cloned()
    }
}

#[cfg(test)]
mod test_splice_database {
    use super::*;

    #[test]
    fn test_splice_database_insert() {
        let mut splice_db = SpliceDatabase::new();
        let splice = splice_db.insert(&[1, 2, 3]);
        assert_eq!(splice.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_splice_database_get() {
        let mut splice_db = SpliceDatabase::new();
        let splice = splice_db.insert(&[1, 2, 3]);
        assert_eq!(splice_db.get(&[1, 2, 3]), Some(splice));
    }

    #[test]
    fn test_splice_database_get_none() {
        let splice_db = SpliceDatabase::new();
        assert_eq!(splice_db.get(&[1, 2, 3]), None);
    }
}
