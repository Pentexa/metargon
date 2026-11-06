//! METARGON v0.8 — bounded posterior of one unknown binary gate mechanism.
//!
//! Knows only a supplied environmental gate likelihood (a model assumption).
//! Learns from gate-open observations, not the simulator's latent rule. This is
//! a small belief tracker, **not** an end-to-end autonomous world model.
use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mechanism { Key, Switch }

#[derive(Clone, Debug)]
pub struct GateModel<const CONTEXTS: usize> {
    switch_prob: [f64; CONTEXTS],
    surprise: [f64; CONTEXTS],
    updates: u64,
}

impl<const CONTEXTS: usize> GateModel<CONTEXTS> {
    pub fn new() -> Result<Self, Error> {
        if CONTEXTS == 0 { return Err(Error::InvalidContext); }
        Ok(Self { switch_prob: [0.5; CONTEXTS],
                  surprise: [0.0; CONTEXTS], updates: 0 })
    }

    /// Called at the beginning of an episode, never using the hidden mode.
    pub fn start(&mut self, context: usize) -> Result<(), Error> {
        let p = *self.switch_prob.get(context).ok_or(Error::InvalidContext)?;
        let surprise = self.surprise[context];
        let hazard = 0.018 + (0.24 * surprise).min(0.23);
        self.switch_prob[context] = p * (1.0 - hazard) + (1.0 - p) * hazard;
        Ok(())
    }

    pub fn posterior(&self, context: usize) -> Result<f64, Error> {
        self.switch_prob.get(context).copied().ok_or(Error::InvalidContext)
    }

    pub fn choose(&self, context: usize) -> Result<Mechanism, Error> {
        Ok(if self.posterior(context)? > 0.5 { Mechanism::Switch } else { Mechanism::Key })
    }

    /// Assumed model: correct gate mechanism opens with probability 0.93.
    /// The simulator can violate this assumption on held-out conditions.
    pub fn predict_open(&self, context: usize, choice: Mechanism) -> Result<f64, Error> {
        let p = self.posterior(context)?;
        Ok(match choice {
            Mechanism::Switch => 0.93 * p + 0.07 * (1.0 - p),
            Mechanism::Key => 0.07 * p + 0.93 * (1.0 - p),
        })
    }

    /// Update only after a real *observed* gate event.
    pub fn observe(&mut self, context: usize, choice: Mechanism, opened: bool)
        -> Result<(), Error>
    {
        let mut prior = self.posterior(context)?;
        let predicted = self.predict_open(context, choice)?;
        let surprise = if opened { 1.0 - predicted } else { predicted };
        self.surprise[context] = 0.8 * self.surprise[context] + 0.2 * surprise;
        if surprise > 0.72 { prior = 0.5 * prior + 0.25; }
        let mut like_switch = if choice == Mechanism::Switch { 0.93 } else { 0.07 };
        let mut like_key = if choice == Mechanism::Key { 0.93 } else { 0.07 };
        if !opened { like_switch = 1.0 - like_switch; like_key = 1.0 - like_key; }
        let den = prior * like_switch + (1.0 - prior) * like_key;
        self.switch_prob[context] = if den > 0.0 { prior * like_switch / den } else { 0.5 };
        self.updates = self.updates.saturating_add(1);
        Ok(())
    }

    pub fn updates(&self) -> u64 { self.updates }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn starts_without_hidden_mode() {
        let mut m = GateModel::<3>::new().unwrap();
        assert_eq!(m.posterior(0), Ok(0.5));
        assert_eq!(m.choose(1), Ok(Mechanism::Key));
        m.start(2).unwrap();
        assert_eq!(m.updates(), 0);
    }
    #[test]
    fn learns_from_public_gate_result() {
        let mut m = GateModel::<2>::new().unwrap();
        let before = m.predict_open(0, Mechanism::Key).unwrap();
        m.observe(0, Mechanism::Key, true).unwrap();
        assert!(m.predict_open(0, Mechanism::Key).unwrap() > before);
        assert_eq!(m.posterior(1), Ok(0.5));
        assert_eq!(m.updates(), 1);
    }
    #[test]
    fn handles_unexpected_failure_and_invalid_context() {
        let mut m = GateModel::<2>::new().unwrap();
        m.observe(0, Mechanism::Key, false).unwrap();
        assert!(m.posterior(0).unwrap() > 0.5);
        assert_eq!(m.predict_open(2, Mechanism::Key), Err(Error::InvalidContext));
        assert_eq!(m.start(2), Err(Error::InvalidContext));
    }
    #[test]
    fn empty_model_refused() { assert!(GateModel::<0>::new().is_err()); }
}
