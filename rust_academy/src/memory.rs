//! Bounded compositional hypothesis memory.

use crate::atoms::Atom;
use std::collections::{BTreeMap, BTreeSet};

/// Hypothesis status (mirrors the Python statuses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// candidate, unsupported
    Hypothesis,
    /// >=1 supporting event, usable for planning
    Predicted,
    /// sufficient support, no contradiction
    Verified,
    /// too many contradictions; excluded from planning
    Contradicted,
}

/// A candidate causal rule: antecedent -> door consequence.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// antecedent atoms (conjunctive)
    pub antecedent: BTreeSet<Atom>,
    /// door id (0/1)
    pub door: u8,
    /// whether the consequence is "opens" (true) or "closes" (false)
    pub opens: bool,
    /// supporting evidence count
    pub sup: u32,
    /// contradicting evidence count
    pub con: u32,
    /// current status
    pub status: Status,
    /// memory step at which it was created / last updated
    pub created_step: u64,
    /// memory step of the last evidence update
    pub last_update: u64,
    /// unique id
    pub id: u64,
}

impl Hypothesis {
    /// Is the hypothesis usable for planning?
    pub fn usable(&self) -> bool {
        self.status != Status::Contradicted
    }

    /// Empirical P(outcome | antecedent) with a weak Beta(1,1) prior.
    pub fn confidence(&self) -> f64 {
        (self.sup as f64 + 1.0) / (self.sup as f64 + self.con as f64 + 2.0)
    }

    /// Deterministic eviction score (mirrors Python `score()`).
    pub fn score(&self) -> f64 {
        if self.status == Status::Contradicted {
            return -1.0 + self.con as f64 * 0.01;
        }
        let mut base = self.confidence();
        match self.status {
            Status::Verified => base += 10.0,
            Status::Predicted => base += 2.0,
            _ => {}
        }
        base - 0.0001 * self.last_update as f64
    }
}

/// Bounded hypothesis memory with deterministic eviction.
#[derive(Debug, Clone)]
pub struct HypothesisMemory {
    /// hard capacity
    pub max_hypotheses: usize,
    /// support needed to verify
    pub verify_support: u32,
    /// contradictions allowed before invalidation
    pub max_contradict: u32,
    /// support needed to become predicted
    pub predict_threshold: u32,
    /// consecutive-true steps counted as one contradiction window
    pub contradict_window: u32,
    hypotheses: Vec<Hypothesis>,
    next_id: u64,
    /// step counter (persists across episodes)
    pub step: u64,
    /// total evictions
    pub evictions: u64,
    /// total generated
    pub generated: u64,
    continuous_true: BTreeMap<u64, u32>,
}

impl Default for HypothesisMemory {
    fn default() -> Self {
        HypothesisMemory {
            max_hypotheses: 128,
            verify_support: 3,
            max_contradict: 2,
            predict_threshold: 1,
            contradict_window: 3,
            hypotheses: Vec::new(),
            next_id: 1,
            step: 0,
            evictions: 0,
            generated: 0,
            continuous_true: BTreeMap::new(),
        }
    }
}

impl HypothesisMemory {
    /// Create with a custom capacity.
    pub fn with_capacity(cap: usize) -> Self {
        let mut m = Self::default();
        m.max_hypotheses = cap;
        m
    }

    /// Current number of stored hypotheses.
    pub fn len(&self) -> usize {
        self.hypotheses.len()
    }

    /// True when the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.hypotheses.is_empty()
    }

    /// Consequence key for a door/opens pair.
    pub fn consequence_key(door: u8, opens: bool) -> String {
        format!("door:D{}:{}", door + 1, if opens { "opens" } else { "closes" })
    }

    /// Query usable hypotheses for a door, sorted by confidence (top first).
    pub fn query(&self, door: u8, opens: Option<bool>) -> Vec<&Hypothesis> {
        let mut out: Vec<&Hypothesis> = self
            .hypotheses
            .iter()
            .filter(|h| h.door == door && h.usable())
            .filter(|h| opens.map(|o| h.opens == o).unwrap_or(true))
            .collect();
        out.sort_by(|a, b| {
            b.confidence()
                .partial_cmp(&a.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.sup.cmp(&a.sup))
        });
        out
    }

    fn find(&self, antecedent: &BTreeSet<Atom>, door: u8, opens: bool)
            -> Option<usize> {
        self.hypotheses
            .iter()
            .position(|h| h.antecedent == *antecedent && h.door == door && h.opens == opens)
    }

    fn add(&mut self, antecedent: BTreeSet<Atom>, door: u8, opens: bool)
           -> Option<usize> {
        if let Some(idx) = self.find(&antecedent, door, opens) {
            return Some(idx);
        }
        if self.hypotheses.len() >= self.max_hypotheses {
            self.evict_one();
        }
        let h = Hypothesis {
            antecedent,
            door,
            opens,
            sup: 0,
            con: 0,
            status: Status::Hypothesis,
            created_step: self.step,
            last_update: self.step,
            id: self.next_id,
        };
        self.next_id += 1;
        self.generated += 1;
        self.hypotheses.push(h);
        Some(self.hypotheses.len() - 1)
    }

    fn evict_one(&mut self) {
        if self.hypotheses.is_empty() {
            return;
        }
        // deterministic lowest-score (ties: oldest last_update, then lowest id)
        let mut victim = 0usize;
        for i in 1..self.hypotheses.len() {
            let a = &self.hypotheses[i];
            let b = &self.hypotheses[victim];
            let better = a.score() < b.score()
                || (a.score() == b.score()
                    && (a.last_update, a.id) < (b.last_update, b.id));
            if better {
                victim = i;
            }
        }
        let id = self.hypotheses[victim].id;
        self.hypotheses.remove(victim);
        self.continuous_true.remove(&id);
        self.evictions += 1;
    }

    fn recompute_status(&mut self, idx: usize) {
        let h = &mut self.hypotheses[idx];
        if h.con > self.max_contradict {
            h.status = Status::Contradicted;
        } else if h.sup >= self.verify_support && h.con <= self.max_contradict {
            h.status = Status::Verified;
        } else if h.sup >= self.predict_threshold {
            h.status = Status::Predicted;
        } else {
            h.status = Status::Hypothesis;
        }
    }

    fn support(&mut self, idx: usize) {
        let h = &mut self.hypotheses[idx];
        h.sup += 1;
        h.last_update = self.step;
        self.continuous_true.remove(&h.id);
        let i = idx;
        drop(h);
        self.recompute_status(i);
    }

    fn contradict(&mut self, idx: usize) {
        let h = &mut self.hypotheses[idx];
        h.con += 1;
        h.last_update = self.step;
        self.continuous_true.remove(&h.id);
        let i = idx;
        drop(h);
        self.recompute_status(i);
    }

    /// Simple candidate generation from an atom set (bounded, like Python).
    fn candidate_antecedents(atoms: &BTreeSet<Atom>, max_candidates: usize)
                             -> Vec<BTreeSet<Atom>> {
        let base: Vec<Atom> = atoms.iter().take(5).cloned().collect();
        let mut cands: Vec<BTreeSet<Atom>> = base.iter().map(|a| BTreeSet::from([*a])).collect();
        'outer: for i in 0..base.len() {
            for j in i + 1..base.len() {
                if cands.len() >= max_candidates {
                    break 'outer;
                }
                cands.push(BTreeSet::from([base[i], base[j]]));
            }
        }
        cands
    }

    /// Record a door event: candidate hypotheses are generated from the
    /// before (and optionally after) atom contexts and supported.
    pub fn observe_event(&mut self, before: &BTreeSet<Atom>,
                          after: Option<&BTreeSet<Atom>>,
                          door: u8, opened: bool) {
        let mut contexts = vec![before.clone()];
        if let Some(a) = after {
            contexts.push(a.clone());
        }
        let mut seen: Vec<BTreeSet<Atom>> = Vec::new();
        for atoms in &contexts {
            for ante in Self::candidate_antecedents(atoms, 10) {
                if seen.contains(&ante) {
                    continue;
                }
                seen.push(ante.clone());
                if let Some(idx) = self.add(ante.clone(), door, opened) {
                    if ante.is_subset(atoms) {
                        self.support(idx);
                    }
                }
            }
        }
    }

    /// Support hypotheses whose antecedent holds while the door stays open.
    pub fn observe_maintain(&mut self, atoms: &BTreeSet<Atom>, door: u8) {
        let mut to_support: Vec<usize> = Vec::new();
        for i in 0..self.hypotheses.len() {
            let h = &self.hypotheses[i];
            if h.door != door || !h.opens || !h.usable() {
                continue;
            }
            if h.antecedent.is_subset(atoms) {
                let c = self.continuous_true.get(&h.id).copied().unwrap_or(0) + 1;
                self.continuous_true.insert(h.id, c);
                if c >= 1 {
                    to_support.push(i);
                }
            } else {
                self.continuous_true.remove(&h.id);
            }
        }
        for i in to_support {
            self.support(i);
        }
    }

    /// Contradict opens-predicting hypotheses whose antecedent holds while
    /// the door stays closed for the window.
    pub fn observe_nonevent(&mut self, atoms: &BTreeSet<Atom>, door: u8) {
        for i in 0..self.hypotheses.len() {
            let h = &self.hypotheses[i];
            if h.door != door || !h.opens || !h.usable() {
                continue;
            }
            if h.antecedent.is_subset(atoms) {
                let c = self.continuous_true.get(&h.id).copied().unwrap_or(0) + 1;
                self.continuous_true.insert(h.id, c);
                if c >= self.contradict_window {
                    let idx = i;
                    self.contradict(idx);
                }
            } else {
                self.continuous_true.remove(&h.id);
            }
        }
    }

    /// Advance the step counter (called once per environment step).
    pub fn step_advance(&mut self) {
        self.step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_supports_candidates() {
        let mut m = HypothesisMemory::default();
        let before = BTreeSet::from([Atom::sw(0, true)]);
        m.observe_event(&before, Some(&before), 0, true);
        let q = m.query(0, Some(true));
        assert!(!q.is_empty());
        assert_eq!(q[0].sup, 1);
    }

    #[test]
    fn maintain_verifies() {
        let mut m = HypothesisMemory::default();
        let atom = BTreeSet::from([Atom::sw(0, true)]);
        m.observe_event(&BTreeSet::new(), Some(&atom), 0, true);
        m.observe_maintain(&atom, 0);
        m.observe_maintain(&atom, 0);
        m.observe_maintain(&atom, 0);
        let q = m.query(0, Some(true));
        assert_eq!(q[0].status, Status::Verified);
    }

    #[test]
    fn nonevent_contradicts() {
        let mut m = HypothesisMemory::default();
        let atom = BTreeSet::from([Atom::sw(0, true)]);
        m.observe_event(&atom, Some(&atom), 0, true);
        for _ in 0..9 {
            m.observe_nonevent(&atom, 0);
        }
        assert!(m.query(0, Some(true)).is_empty());
    }

    #[test]
    fn bounded_eviction() {
        let mut m = HypothesisMemory::with_capacity(5);
        for i in 0..20u8 {
            let ant = BTreeSet::from([
                Atom::sw(0, i & 1 == 1),
                Atom::sw(1, i & 2 == 2),
                Atom::empty(if i % 3 == 0 { 0 } else { 1 }),
            ]);
            m.add(ant, 0, true);
        }
        assert!(m.len() <= 5);
        assert!(m.evictions > 0);
    }

    #[test]
    fn score_ordering() {
        let mut m = HypothesisMemory::default();
        let a = BTreeSet::from([Atom::sw(0, true)]);
        let b = BTreeSet::from([Atom::sw(1, true)]);
        m.add(a.clone(), 0, true);
        m.add(b.clone(), 0, true);
        m.support(0);
        m.support(0);
        assert!(m.hypotheses[0].score() > m.hypotheses[1].score());
    }
}
