//! Conjunctive and disjunctive rule evaluation over the atom vocabulary.

use crate::atoms::Atom;
use std::collections::BTreeSet;

/// A conjunctive rule: all atoms must hold.
pub type Rule = BTreeSet<Atom>;

/// A disjunctive rule: any clause (conjunctive rule) suffices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnyRule(pub Vec<Rule>);

/// A snapshot of the public atoms that currently hold.
#[derive(Debug, Clone, Default)]
pub struct WorldState {
    /// switch state by switch id
    pub switches: [bool; 2],
    /// plate load: None = empty, Some(0/1) = object id
    pub plates: [Option<u8>; 2],
}

impl WorldState {
    /// All currently-true grounded atoms.
    pub fn atoms(&self) -> BTreeSet<Atom> {
        let mut out = BTreeSet::new();
        for (i, v) in self.switches.iter().enumerate() {
            out.insert(Atom::sw(i as u8, *v));
        }
        for (i, load) in self.plates.iter().enumerate() {
            match load {
                Some(obj) => {
                    out.insert(Atom::load(i as u8, *obj));
                    out.insert(Atom::load_any(i as u8));
                }
                None => {
                    out.insert(Atom::empty(i as u8));
                }
            }
        }
        out
    }

    /// Whether a single atom holds.
    pub fn atom_holds(&self, a: &Atom) -> bool {
        match a.kind {
            crate::AtomKind::Switch => {
                a.subject < 2 && self.switches[a.subject as usize] == a.value
            }
            crate::AtomKind::Load => {
                a.subject < 2 && self.plates[a.subject as usize] == Some(if a.value { 1 } else { 0 })
            }
            crate::AtomKind::LoadAny => {
                a.subject < 2 && self.plates[a.subject as usize].is_some()
            }
            crate::AtomKind::Empty => {
                a.subject < 2 && self.plates[a.subject as usize].is_none()
            }
        }
    }

    /// Whether a conjunctive rule holds.
    pub fn rule_holds(&self, rule: &Rule) -> bool {
        rule.iter().all(|a| self.atom_holds(a))
    }

    /// Whether a disjunctive rule holds.
    pub fn any_rule_holds(&self, rule: &AnyRule) -> bool {
        rule.0.iter().any(|clause| self.rule_holds(clause))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atoms::Atom;

    #[test]
    fn conjunctive_requires_all() {
        let mut st = WorldState::default();
        st.switches[0] = true;
        let rule: Rule = BTreeSet::from([Atom::sw(0, true), Atom::empty(0)]);
        assert!(st.rule_holds(&rule));
        st.plates[0] = Some(0);
        assert!(!st.rule_holds(&rule));
    }

    #[test]
    fn disjunctive_any_clause() {
        let mut st = WorldState::default();
        st.switches[0] = true;
        let rule = AnyRule(vec![
            BTreeSet::from([Atom::sw(0, false)]),
            BTreeSet::from([Atom::sw(1, true)]),
            BTreeSet::from([Atom::sw(0, true), Atom::empty(1)]),
        ]);
        assert!(st.any_rule_holds(&rule));
    }
}
