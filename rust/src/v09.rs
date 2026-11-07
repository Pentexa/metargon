//! METARGON v0.9 — compositional causal memory kernel (Phase C/F).
//!
//! A bounded, allocation-free-in-steady-state hypothesis store for the
//! "Mechanism Lab" research environment. It stores conjunctive causal
//! hypotheses of the form *antecedent (grounded atoms over
//! plates/switches) -> door outcome*, distinguishing observation,
//! hypothesis, predicted, verified and contradicted statuses.
//! Hypotheses never become verified facts until enough supporting
//! evidence accumulates; contradicting evidence demotes or invalidates
//! them. Memory is bounded by a compile-time capacity with a
//! deterministic score-based eviction policy.
//!
//! This Rust kernel is deliberately narrower than the Python reference:
//! it implements the memory itself (the core research contribution), not
//! the full planner. It is `#![forbid(unsafe_code)]` safe.

/// Atom kinds over the lab's public components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    /// ("load", plate, obj): plate carries exactly that object.
    Load { plate: u8, obj: u8 },
    /// ("loadany", plate): plate carries any object.
    LoadAny { plate: u8 },
    /// ("sw", switch, value): switch is in that state.
    Switch { switch: u8, value: bool },
}

/// A door outcome (consequence of a hypothesis).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Opens { door: u8 },
    Closes { door: u8 },
}

/// Lifecycle status of a hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Hypothesis,
    Predicted,
    Verified,
    Contradicted,
}

/// A single stored hypothesis.
#[derive(Clone, Copy, Debug)]
pub struct Hypothesis {
    pub antecedent: [Atom; MAX_ATOMS],
    pub n_atoms: usize,
    pub consequence: Outcome,
    pub support: u16,
    pub contradiction: u16,
    pub status: Status,
    pub last_update: u64,
    pub id: u64,
}

/// Maximum antecedent size (conjunctive hypotheses are small).
pub const MAX_ATOMS: usize = 2;
/// Maximum number of stored hypotheses (hard budget).
pub const MAX_HYPOTHESES: usize = 128;
/// Supporting events before a hypothesis is verified.
pub const VERIFY_SUPPORT: u16 = 3;
/// Contradicting events that invalidate.
pub const MAX_CONTRADICT: u16 = 2;
/// Support required to be "predicted".
pub const PREDICT_THRESHOLD: u16 = 1;
/// Deterministic tie-break penalty per step of age.
pub const AGE_PENALTY: f64 = 0.0001;

/// Bounded compositional causal memory.
pub struct CausalMemory {
    hypotheses: [Option<Hypothesis>; MAX_HYPOTHESES],
    next_id: u64,
    step: u64,
    evictions: u64,
    generated: u64,
    /// Continuous-true step counters for nonevent contradiction detection.
    continuous_true: [u16; MAX_HYPOTHESES],
}

impl CausalMemory {
    pub fn new() -> Self {
        Self {
            hypotheses: [None; MAX_HYPOTHESES],
            next_id: 1,
            step: 0,
            evictions: 0,
            generated: 0,
            continuous_true: [0; MAX_HYPOTHESES],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.hypotheses.iter().filter(|h| h.is_some()).count()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn step_advance(&mut self) {
        self.step = self.step.wrapping_add(1);
    }

    fn score(&self, h: &Hypothesis) -> f64 {
        if h.status == Status::Contradicted {
            return -1.0 + f64::from(h.contradiction) * 0.01;
        }
        let mut base = f64::from(h.support + 1) / f64::from(h.support + h.contradiction + 2);
        match h.status {
            Status::Verified => base += 10.0,
            Status::Predicted => base += 2.0,
            _ => {}
        }
        base - AGE_PENALTY * (self.step.wrapping_sub(h.last_update)) as f64
    }

    /// Deterministic eviction: lowest score; ties -> oldest.
    fn evict_one(&mut self) {
        let mut victim: Option<usize> = None;
        let mut victim_score = f64::INFINITY;
        for i in 0..MAX_HYPOTHESES {
            if let Some(h) = &self.hypotheses[i] {
                let s = self.score(h);
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
            support: 0,
            contradiction: 0,
            status: Status::Hypothesis,
            last_update: self.step,
            id,
        });
        self.generated = self.generated.wrapping_add(1);
        Some(i)
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

    /// Observe a door event (opened or closed) whose context was `ctx`.
    /// Supports matching hypotheses; generates singleton/pair candidates.
    pub fn observe_event(&mut self, ctx: &[Atom], outcome: Outcome) {
        let n = ctx.len().min(MAX_ATOMS);
        let mut candidates: [[Atom; MAX_ATOMS]; 4] =
            [[Atom::Switch { switch: 0, value: false }; MAX_ATOMS]; 4];
        let mut lens = [0usize; 4];
        let mut n_cand = 0;
        // singletons
        for &a in ctx.iter().take(2) {
            if n_cand < 4 {
                candidates[n_cand][0] = a;
                lens[n_cand] = 1;
                n_cand += 1;
            }
        }
        // pairs (only when the context has >= 2 atoms)
        if n >= 2 && n_cand < 4 {
            candidates[n_cand][0] = ctx[0];
            candidates[n_cand][1] = ctx[1];
            lens[n_cand] = 2;
            n_cand += 1;
        }
        for c in 0..n_cand {
            if let Some(i) = self.add(&candidates[c][..lens[c]], outcome) {
                let h = self.hypotheses[i].as_mut().unwrap();
                h.support = h.support.saturating_add(1);
                h.last_update = self.step;
                self.continuous_true[i] = 0;
                Self::recompute_status(h);
            }
        }
    }

    /// Observe that a door did *not* produce the expected outcome while its
    /// antecedent stayed true for `window` consecutive steps.
    pub fn observe_nonevent(&mut self, ctx: &[Atom], outcome: Outcome, window: u16) {
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
                    if self.continuous_true[i] >= window {
                        let h = self.hypotheses[i].as_mut().unwrap();
                        h.contradiction = h.contradiction.saturating_add(1);
                        h.last_update = self.step;
                        self.continuous_true[i] = 0;
                        Self::recompute_status(h);
                    }
                } else {
                    self.continuous_true[i] = 0;
                }
            }
        }
    }

    /// Query usable hypotheses for a door, best confidence first.
    pub fn query(&self, outcome: Outcome) -> Vec<Hypothesis> {
        let mut out: Vec<Hypothesis> = self
            .hypotheses
            .iter()
            .filter_map(|h| h.as_ref())
            .filter(|h| h.consequence == outcome && h.status != Status::Contradicted)
            .copied()
            .collect();
        out.sort_by(|a, b| {
            let ca = f64::from(a.support + 1) / f64::from(a.support + a.contradiction + 2);
            let cb = f64::from(b.support + 1) / f64::from(b.support + b.contradiction + 2);
            cb.partial_cmp(&ca).unwrap_or(std::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        out
    }

    pub fn diagnostics(&self) -> (usize, u64, u64, [usize; 4]) {
        let mut counts = [0usize; 4];
        for h in self.hypotheses.iter().flatten() {
            counts[match h.status {
                Status::Hypothesis => 0,
                Status::Predicted => 1,
                Status::Verified => 2,
                Status::Contradicted => 3,
            }] += 1;
        }
        (self.len(), self.evictions, self.generated, counts)
    }
}

impl Default for CausalMemory {
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
    fn bounded_and_status_lifecycle() {
        let mut m = CausalMemory::new();
        let c = ctx_load_sw();
        let out = Outcome::Opens { door: 0 };
        for _ in 0..VERIFY_SUPPORT - 1 {
            m.observe_event(&c, out);
            m.step_advance();
        }
        assert!(!m.query(out).iter().any(|h| h.status == Status::Verified));
        m.observe_event(&c, out);
        assert!(m.query(out).iter().any(|h| h.status == Status::Verified));
        assert_eq!(m.len(), 3); // two singletons + one pair
    }

    #[test]
    fn contradiction_invalidates() {
        let mut m = CausalMemory::new();
        let c = [Atom::Switch { switch: 1, value: true }];
        let out = Outcome::Opens { door: 1 };
        for _ in 0..VERIFY_SUPPORT {
            m.observe_event(&c, out);
            m.step_advance();
        }
        assert!(m.query(out).iter().any(|h| h.status == Status::Verified));
        for _ in 0..(MAX_CONTRADICT + 1) * 3 {
            m.observe_nonevent(&c, out, 3);
            m.step_advance();
        }
        assert!(m.query(out).is_empty());
        let (_, _, _, counts) = m.diagnostics();
        assert!(counts[3] >= 1); // contradicted bucket non-empty
    }

    #[test]
    fn eviction_is_bounded_and_deterministic() {
        let mut m1 = CausalMemory::new();
        let mut m2 = CausalMemory::new();
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
        let d1 = m1.diagnostics();
        let d2 = m2.diagnostics();
        assert_eq!(d1, d2);
    }

    #[test]
    fn query_orders_by_confidence() {
        let mut m = CausalMemory::new();
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
