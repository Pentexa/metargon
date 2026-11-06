//! Atom vocabulary and bounded candidate generation.

use std::collections::BTreeSet;

/// Atom kind, mirroring the Python `("sw" | "load" | "loadany" | "empty", ...)`
/// vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomKind {
    /// ("sw", switch, value)
    Switch,
    /// ("load", plate, object)
    Load,
    /// ("loadany", plate)
    LoadAny,
    /// ("empty", plate)
    Empty,
}

/// A grounded atom over the public vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Atom {
    /// Atom kind.
    pub kind: AtomKind,
    /// Subject index: switch or plate id (0/1).
    pub subject: u8,
    /// Boolean value (switch state or object id 0/1 for Load).
    pub value: bool,
}

impl Atom {
    /// Build a switch atom ("sw", subject, value).
    pub fn sw(subject: u8, value: bool) -> Self {
        Atom { kind: AtomKind::Switch, subject, value }
    }
    /// Build a load atom ("load", plate, object).
    pub fn load(plate: u8, object: u8) -> Self {
        Atom { kind: AtomKind::Load, subject: plate, value: object != 0 }
    }
    /// Build a "loadany" atom.
    pub fn load_any(plate: u8) -> Self {
        Atom { kind: AtomKind::LoadAny, subject: plate, value: false }
    }
    /// Build an "empty" atom.
    pub fn empty(plate: u8) -> Self {
        Atom { kind: AtomKind::Empty, subject: plate, value: false }
    }
}

/// The fixed probe vocabulary (mirrors `VOCAB_SINGLES` in Python):
/// both polarities of every switch, plus load/loadany/empty for every plate.
pub fn vocabulary_singles(n_switches: u8, n_plates: u8) -> Vec<Atom> {
    let mut out = Vec::new();
    for s in 0..n_switches {
        out.push(Atom::sw(s, true));
        out.push(Atom::sw(s, false));
    }
    for p in 0..n_plates {
        out.push(Atom::empty(p));
        out.push(Atom::load_any(p));
        out.push(Atom::load(p, 0));
        out.push(Atom::load(p, 1));
    }
    out
}

/// Bounded candidate generation over the fixed vocabulary (singletons and
/// pairs), deterministically ordered — mirrors the Python probe builder.
pub struct Candidates<'a> {
    singles: &'a [Atom],
    tried: &'a BTreeSet<BTreeSet<Atom>>,
    max_pairs: usize,
}

impl<'a> Candidates<'a> {
    /// Build a candidate generator over `singles` (vocabulary) skipping
    /// antecedents already in `tried`.
    pub fn new(singles: &'a [Atom], tried: &'a BTreeSet<BTreeSet<Atom>>,
               max_pairs: usize) -> Self {
        Candidates { singles, tried, max_pairs }
    }

    /// All novel singleton antecedents.
    pub fn singles(&self) -> Vec<BTreeSet<Atom>> {
        self.singles
            .iter()
            .map(|a| BTreeSet::from([*a]))
            .filter(|s| !self.tried.contains(s))
            .collect()
    }

    /// Bounded novel pairs (deterministic order).
    pub fn pairs(&self) -> Vec<BTreeSet<Atom>> {
        let mut out = Vec::new();
        for i in 0..self.singles.len() {
            for j in i + 1..self.singles.len() {
                let mut p = BTreeSet::new();
                p.insert(self.singles[i]);
                p.insert(self.singles[j]);
                if !self.tried.contains(&p) {
                    out.push(p);
                    if out.len() >= self.max_pairs {
                        return out;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_size() {
        assert_eq!(vocabulary_singles(2, 2).len(), 2 * 2 + 2 * 4);
    }

    #[test]
    fn candidates_skip_tried() {
        let vocab = vocabulary_singles(1, 1);
        let mut tried = BTreeSet::new();
        tried.insert(BTreeSet::from([Atom::sw(0, true)]));
        let c = Candidates::new(&vocab, &tried, 8);
        assert!(!c.singles().contains(&BTreeSet::from([Atom::sw(0, true)])));
        assert!(c.singles().contains(&BTreeSet::from([Atom::sw(0, false)])));
    }
}
