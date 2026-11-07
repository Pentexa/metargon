//! METARGON v0.10 — compositional causal memory kernel with evidence decay
//! and prioritized replay (Rust mirror of python/compositional_memory_v10.py).
//!
//! Changes vs v0.9 kernel (`v09.rs`):
//!   * support/contradiction are f64 weights (EWMA-style) rather than u16
//!     counts; every `episode_end()` decays all evidence by `EVENT_DECAY`
//!     so mechanism changes slowly unlearn old rules;
//!   * candidate generation is complete for the event context (all
//!     singletons + all pairs) up to a cap, instead of a fixed 2-atom
//!     slice;
//!   * informative causal events are stored in a bounded replay buffer with
//!     priority 1/(1+support); the top-priority older entries are replayed
//!     after each event (strengthening sparse evidence);
//!   * status thresholds use the weighted values.
//!
//! The kernel is `#![forbid(unsafe_code)]`-safe and keeps its steady-state
//! storage in fixed arrays (no heap allocation below operation loops).

use crate::v09::{Atom, Outcome, Status};

/// Hard memory budget (same as Python).
pub const MAX_HYPOTHESES: usize = 128;
/// Weighted support before a hypothesis is verified.
pub const VERIFY_SUPPORT: f64 = 3.0;
/// Weighted contradiction before a hypothesis is contradicted.
pub const MAX_CONTRADICT: f64 = 2.0;
/// Weighted support to be "predicted".
pub const PREDICT_THRESHOLD: f64 = 1.0;
/// Consecutive believed-true steps without the predicted effect count as
/// one contradiction.
pub const CONTRADICT_WINDOW: u16 = 3;
/// Per-episode evidence decay.
pub const EVENT_DECAY: f64 = 0.70;
/// Replay buffer capacity.
pub const REPLAY_MAX: usize = 64;
/// Replay applications per event.
pub const REPLAY_PER_EVENT: usize = 8;
/// Candidate-generation cap per event.
pub const CANDIDATES_PER_EVENT: usize = 32;
/// Maximum antecedent size (conjunctive hypotheses are small).
pub const MAX_ATOMS: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct Hypothesis {
    pub antecedent: [Atom; MAX_ATOMS],
    pub n_atoms: usize,
    pub consequence: Outcome,
    pub support: f64,
    pub contradiction: f64,
    pub n_sup: u64,
    pub n_con: u64,
    pub n_replay: u64,
    pub status: Status,
    pub last_update: u64,
    pub id: u64,
}

impl Hypothesis {
    pub fn confidence(&self) -> f64 {
        (self.support + 1.0) / (self.support + self.contradiction + 2.0)
    }

    fn score(&self, step: u64) -> f64 {
        if self.status == Status::Contradicted {
            return -1.0 + self.contradiction.min(10.0) * 0.01;
        }
        let mut base = self.confidence();
        match self.status {
            Status::Verified => base += 10.0,
            Status::Predicted => base += 2.0,
            _ => {}
        }
        base - 0.0001 * step.wrapping_sub(self.last_update) as f64
    }
}

#[derive(Clone, Copy, Debug)]
struct ReplayEntry {
    priority: f64,
    antecedent: [Atom; MAX_ATOMS],
    n_atoms: usize,
    consequence: Outcome,
}

/// Bounded compositional causal memory (v0.10 semantics).
pub struct CausalMemoryV10 {
    hypotheses: [Option<Hypothesis>; MAX_HYPOTHESES],
    replay: [Option<ReplayEntry>; REPLAY_MAX],
    replay_len: usize,
    next_id: u64,
    step: u64,
    evictions: u64,
    generated: u64,
    replay_applied: u64,
    continuous_true: [u16; MAX_HYPOTHESES],
}

impl CausalMemoryV10 {
    pub fn new() -> Self {
        Self {
            hypotheses: [None; MAX_HYPOTHESES],
            replay: [None; REPLAY_MAX],
            replay_len: 0,
            next_id: 1,
            step: 0,
            evictions: 0,
            generated: 0,
            replay_applied: 0,
            continuous_true: [0; MAX_HYPOTHESES],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.hypotheses.iter().filter(|h| h.is_some()).count()
    }

    pub fn step_advance(&mut self) {
        self.step = self.step.wrapping_add(1);
    }

    /// Decay all evidence by EVENT_DECAY (called once per episode boundary).
    pub fn episode_end(&mut self) {
        for slot in self.hypotheses.iter_mut() {
            if let Some(h) = slot {
                h.support *= EVENT_DECAY;
                h.contradiction *= EVENT_DECAY;
                Self::recompute_status(h);
            }
        }
        for c in self.continuous_true.iter_mut() {
            *c = c.saturating_sub(1);
        }
    }

    fn recompute_status(h: &mut Hypothesis) {
        h.status = if h.contradiction > MAX_CONTRADICT {
            Status::Contradicted
        } else if h.support >= VERIFY_SUPPORT {
            Status::Verified
        } else if h.support >= PREDICT_THRESHOLD {
            Status::Predicted
        } else {
            Status::Hypothesis
        };
    }

    fn find(&self, antecedent: &[Atom], consequence: Outcome) -> Option<usize> {
        self.hypotheses.iter().position(|h| {
            if let Some(h) = h {
                h.consequence == consequence && h.n_atoms == antecedent.len() && {
                    let mut ok = true;
                    for (i, a) in antecedent.iter().enumerate() {
                        if h.antecedent[i] != *a {
                            ok = false;
                            break;
                        }
                    }
                    ok
                }
            } else {
                false
            }
        })
    }

    fn add(&mut self, antecedent: &[Atom], consequence: Outcome) -> Option<usize> {
        if let Some(i) = self.find(antecedent, consequence) {
            return Some(i);
        }
        if self.len() >= MAX_HYPOTHESES {
            self.evict_one();
        }
        let mut idx = None;
        for i in 0..MAX_HYPOTHESES {
            if self.hypotheses[i].is_none() {
                idx = Some(i);
                break;
            }
        }
        let i = idx?;
        let mut ant = [Atom::Switch { switch: 0, value: false }; MAX_ATOMS];
        for (k, a) in antecedent.iter().enumerate() {
            ant[k] = *a;
        }
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.hypotheses[i] = Some(Hypothesis {
            antecedent: ant,
            n_atoms: antecedent.len(),
            consequence,
            support: 0.0,
            contradiction: 0.0,
            n_sup: 0,
            n_con: 0,
            n_replay: 0,
            status: Status::Hypothesis,
            last_update: self.step,
            id,
        });
        self.generated = self.generated.wrapping_add(1);
        Some(i)
    }

    fn evict_one(&mut self) {
        let mut victim: Option<usize> = None;
        let mut victim_score = f64::INFINITY;
        for i in 0..MAX_HYPOTHESES {
            if let Some(h) = &self.hypotheses[i] {
                let s = h.score(self.step);
                if s < victim_score
                    || (s - victim_score).abs() < 1e-12
                        && victim.map_or(true, |v| {
                            let a = self.hypotheses[v].as_ref().unwrap();
                            h.last_update < a.last_update
                                || (h.last_update == a.last_update && h.id < a.id)
                        })
                {
                    victim = Some(i);
                    victim_score = s;
                }
            }
        }
        if let Some(i) = victim {
            self.hypotheses[i] = None;
            self.continuous_true[i] = 0;
            self.evictions = self.evictions.wrapping_add(1);
        }
    }

    fn support(&mut self, i: usize, from_replay: bool) {
        if let Some(h) = &mut self.hypotheses[i] {
            h.support += 1.0;
            h.n_sup += 1;
            if from_replay {
                h.n_replay += 1;
            }
            h.last_update = self.step;
            self.continuous_true[i] = 0;
            Self::recompute_status(h);
        }
    }

    fn contradict(&mut self, i: usize) {
        if let Some(h) = &mut self.hypotheses[i] {
            h.contradiction += 1.0;
            h.n_con += 1;
            h.last_update = self.step;
            self.continuous_true[i] = 0;
            Self::recompute_status(h);
        }
    }

    fn replay_priority(&self, antecedent: &[Atom], consequence: Outcome) -> f64 {
        match self.find(antecedent, consequence) {
            None => 1.0,
            Some(i) => {
                let sup = self.hypotheses[i].as_ref().unwrap().support.max(0.0);
                1.0 / (1.0 + sup)
            }
        }
    }

    fn store_replay(&mut self, antecedent: &[Atom], consequence: Outcome) {
        let priority = self.replay_priority(antecedent, consequence);
        let mut ant = [Atom::Switch { switch: 0, value: false }; MAX_ATOMS];
        for (k, a) in antecedent.iter().enumerate() {
            ant[k] = *a;
        }
        let entry = ReplayEntry {
            priority,
            antecedent: ant,
            n_atoms: antecedent.len(),
            consequence,
        };
        if self.replay_len < REPLAY_MAX {
            self.replay[self.replay_len] = Some(entry);
            self.replay_len += 1;
        } else {
            // drop the lowest-priority entry
            let mut worst = 0;
            let mut worst_p = f64::INFINITY;
            for k in 0..REPLAY_MAX {
                if let Some(e) = &self.replay[k] {
                    if e.priority < worst_p {
                        worst_p = e.priority;
                        worst = k;
                    }
                }
            }
            if entry.priority > worst_p {
                self.replay[worst] = Some(entry);
            }
        }
    }

    fn apply_replay(&mut self, items: &[ReplayEntry]) {
        let mut sorted: Vec<ReplayEntry> = items.to_vec();
        sorted.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        let mut taken = 0;
        for e in sorted.iter() {
            if taken >= REPLAY_PER_EVENT {
                break;
            }
            let i = match self.find(&e.antecedent[..e.n_atoms], e.consequence) {
                Some(i) => i,
                None => match self.add(&e.antecedent[..e.n_atoms], e.consequence) {
                    Some(i) => i,
                    None => continue,
                },
            };
            self.support(i, true);
            self.replay_applied = self.replay_applied.wrapping_add(1);
            taken += 1;
        }
    }

    /// Observe a door event whose context was `ctx` (atoms believed true).
    /// Supports matching hypotheses and generates singleton/pair candidates.
    pub fn observe_event(&mut self, ctx: &[Atom], outcome: Outcome) {
        let old_len = self.replay_len;
        let old_entries: Vec<ReplayEntry> = self.replay[..old_len].iter().flatten().copied().collect();
        let n = ctx.len();
        let mut candidates: Vec<[Atom; MAX_ATOMS]> = Vec::with_capacity(CANDIDATES_PER_EVENT);
        let mut lens: Vec<usize> = Vec::with_capacity(CANDIDATES_PER_EVENT);
        for &a in ctx.iter() {
            if candidates.len() >= CANDIDATES_PER_EVENT {
                break;
            }
            let mut ant = [Atom::Switch { switch: 0, value: false }; MAX_ATOMS];
            ant[0] = a;
            candidates.push(ant);
            lens.push(1);
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if candidates.len() >= CANDIDATES_PER_EVENT {
                    break;
                }
                let mut ant = [Atom::Switch { switch: 0, value: false }; MAX_ATOMS];
                ant[0] = ctx[i];
                ant[1] = ctx[j];
                candidates.push(ant);
                lens.push(2);
            }
            if candidates.len() >= CANDIDATES_PER_EVENT {
                break;
            }
        }
        for c in 0..candidates.len() {
            if let Some(i) = self.add(&candidates[c][..lens[c]], outcome) {
                self.support(i, false);
            }
            self.store_replay(&candidates[c][..lens[c]], outcome);
        }
        self.apply_replay(&old_entries);
    }

    /// Observe that a door did not produce the expected outcome while its
    /// antecedent stayed true for CONTRADICT_WINDOW consecutive steps.
    pub fn observe_nonevent(&mut self, ctx: &[Atom], outcome: Outcome) {
        for i in 0..MAX_HYPOTHESES {
            if let Some(h) = &self.hypotheses[i] {
                if h.consequence != outcome || h.status == Status::Contradicted {
                    continue;
                }
                let sat = h.n_atoms <= ctx.len() && {
                    let mut ok = true;
                    for k in 0..h.n_atoms {
                        if !ctx.contains(&h.antecedent[k]) {
                            ok = false;
                            break;
                        }
                    }
                    ok
                };
                if sat {
                    self.continuous_true[i] = self.continuous_true[i].saturating_add(1);
                    if self.continuous_true[i] >= CONTRADICT_WINDOW {
                        self.contradict(i);
                    }
                } else {
                    self.continuous_true[i] = 0;
                }
            }
        }
    }

    /// Query usable hypotheses for a door outcome, best confidence first.
    pub fn query(&self, outcome: Outcome) -> Vec<Hypothesis> {
        let mut out: Vec<Hypothesis> = self
            .hypotheses
            .iter()
            .filter_map(|h| h.as_ref())
            .filter(|h| h.consequence == outcome && h.status != Status::Contradicted)
            .copied()
            .collect();
        out.sort_by(|a, b| {
            b.confidence()
                .partial_cmp(&a.confidence())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        out
    }

    pub fn diagnostics(&self) -> (usize, u64, u64, u64, usize, [usize; 4], f64) {
        let mut counts = [0usize; 4];
        let mut total_support = 0.0;
        for h in self.hypotheses.iter().flatten() {
            counts[match h.status {
                Status::Hypothesis => 0,
                Status::Predicted => 1,
                Status::Verified => 2,
                Status::Contradicted => 3,
            }] += 1;
            total_support += h.support;
        }
        (
            self.len(),
            self.evictions,
            self.generated,
            self.replay_applied,
            self.replay_len,
            counts,
            total_support,
        )
    }
}

impl Default for CausalMemoryV10 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_load_sw() -> [Atom; 2] {
        [
            Atom::Load { plate: 1, obj: 0 },
            Atom::Switch { switch: 1, value: true },
        ]
    }

    #[test]
    fn status_lifecycle_weighted() {
        let mut m = CausalMemoryV10::new();
        let c = ctx_load_sw();
        let out = Outcome::Opens { door: 0 };
        m.observe_event(&c, out);
        let h = m.query(out)[0];
        assert_eq!(h.status, Status::Predicted);
        m.observe_event(&c, out);
        m.observe_event(&c, out);
        let h = m.query(out)[0];
        assert!(h.support >= VERIFY_SUPPORT);
        assert_eq!(h.status, Status::Verified);
    }

    #[test]
    fn contradiction_demotes_and_invalidates() {
        let mut m = CausalMemoryV10::new();
        let c = [Atom::Switch { switch: 1, value: true }];
        let out = Outcome::Opens { door: 1 };
        m.observe_event(&c, out);
        for _ in 0..CONTRADICT_WINDOW {
            m.observe_nonevent(&c, out);
        }
        assert!(m.query(out).iter().any(|h| h.contradiction == 1.0));
        for _ in 0..CONTRADICT_WINDOW {
            m.observe_nonevent(&c, out);
        }
        for _ in 0..CONTRADICT_WINDOW {
            m.observe_nonevent(&c, out);
        }
        assert!(m.query(out).is_empty());
        let (_, _, _, _, _, counts, _) = m.diagnostics();
        assert!(counts[3] >= 1);
    }

    #[test]
    fn decay_reduces_support() {
        let mut m = CausalMemoryV10::new();
        let c = [Atom::Switch { switch: 1, value: true }];
        let out = Outcome::Opens { door: 0 };
        m.observe_event(&c, out);
        let before = m.query(out)[0].support;
        m.episode_end();
        let after = m.query(out)[0].support;
        assert!(after < before);
        assert!(after > 0.0);
    }

    #[test]
    fn replay_prioritizes_and_is_bounded() {
        let mut m = CausalMemoryV10::new();
        let c1 = [Atom::Switch { switch: 1, value: true }];
        let c2 = [Atom::Switch { switch: 2, value: false }];
        let out = Outcome::Opens { door: 0 };
        m.observe_event(&c1, out);
        let applied_after_first = m.diagnostics().3;
        m.observe_event(&c2, out);
        assert!(m.diagnostics().3 > applied_after_first);
        assert!(m.diagnostics().4 <= REPLAY_MAX);
        // fill beyond capacity
        for k in 0..200u8 {
            let c = [Atom::Switch { switch: 1, value: k % 2 == 0 }];
            m.observe_event(&c, out);
        }
        assert!(m.diagnostics().4 <= REPLAY_MAX);
    }

    #[test]
    fn eviction_deterministic() {
        let mut m1 = CausalMemoryV10::new();
        let mut m2 = CausalMemoryV10::new();
        for round in 0..10u8 {
            let ctx = [
                Atom::Switch { switch: 1, value: round % 2 == 0 },
                Atom::Load { plate: 1, obj: round % 2 },
            ];
            m1.observe_event(&ctx, Outcome::Opens { door: 0 });
            m1.step_advance();
            m2.observe_event(&ctx, Outcome::Opens { door: 0 });
            m2.step_advance();
        }
        assert!(m1.len() <= MAX_HYPOTHESES);
        assert_eq!(m1.diagnostics(), m2.diagnostics());
    }

    #[test]
    fn query_orders_by_confidence() {
        let mut m = CausalMemoryV10::new();
        let out = Outcome::Opens { door: 0 };
        let a = [Atom::Switch { switch: 1, value: true }];
        let b = [Atom::Switch { switch: 2, value: true }];
        for _ in 0..4 {
            m.observe_event(&a, out);
        }
        m.observe_event(&b, out);
        let q = m.query(out);
        assert!(!q.is_empty());
        assert_eq!(q[0].antecedent[0], a[0]);
    }
}
