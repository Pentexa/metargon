//! CORVID v0.7 bounded one-step diagnostic response predictor.
//!
//! This kernel learns the distribution of observations conditional on public
//! context and chosen action, NOT the truth value. It is intentionally not a
//! complete world simulator. No heap allocation, no I/O, no hidden state.
use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action { Passive, Probe }
impl Action { fn slot(self) -> usize { match self { Self::Passive => 0, Self::Probe => 1 } } }

#[derive(Clone, Debug)]
pub struct ResponseMemory<const CONTEXTS: usize, const WINDOW: usize> {
    observations: [[[[u8; WINDOW]; 2]; 2]; CONTEXTS],
    counts: [[[usize; 2]; 2]; CONTEXTS],
    cursors: [[[usize; 2]; 2]; CONTEXTS],
    updates: u64,
}

impl<const CONTEXTS: usize, const WINDOW: usize> ResponseMemory<CONTEXTS, WINDOW> {
    pub fn new() -> Result<Self, Error> {
        if CONTEXTS == 0 || WINDOW == 0 { return Err(Error::InvalidContext); }
        Ok(Self {
            observations: [[[[0; WINDOW]; 2]; 2]; CONTEXTS],
            counts: [[[0; 2]; 2]; CONTEXTS],
            cursors: [[[0; 2]; 2]; CONTEXTS],
            updates: 0,
        })
    }

    /// A publicly observed diagnostic bit, not an evaluator label.
    pub fn observe(&mut self, context: usize, first: bool, action: Action,
                   readout: bool) -> Result<(), Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        let f = usize::from(first);
        let a = action.slot();
        let slot = self.cursors[context][f][a];
        self.observations[context][f][a][slot] = u8::from(readout);
        self.cursors[context][f][a] = (slot + 1) % WINDOW;
        self.counts[context][f][a] = (self.counts[context][f][a] + 1).min(WINDOW);
        self.updates = self.updates.saturating_add(1);
        Ok(())
    }

    /// Beta(1,1) predictive probability of observing readout=1. The Python
    /// reference additionally pools across contexts; Rust is a bounded kernel.
    pub fn predict_readout_one(&self, context: usize, first: bool,
                               action: Action) -> Result<f64, Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        let f = usize::from(first);
        let a = action.slot();
        let n = self.counts[context][f][a];
        let ones: usize = self.observations[context][f][a]
            .iter().take(n).map(|b| usize::from(*b)).sum();
        Ok((1.0 + ones as f64) / (2.0 + n as f64))
    }

    /// Expected correctness of a binary action's two hypothetical outcomes,
    /// given externally supplied calibrated truth posteriors for each readout.
    pub fn expected_accuracy(p_read_one: f64, p_truth_if_zero: f64,
                             p_truth_if_one: f64) -> Result<f64, Error> {
        let values = [p_read_one, p_truth_if_zero, p_truth_if_one];
        if values.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
            return Err(Error::InvalidProbability);
        }
        Ok((1.0 - p_read_one) * p_truth_if_zero.max(1.0 - p_truth_if_zero)
            + p_read_one * p_truth_if_one.max(1.0 - p_truth_if_one))
    }

    pub fn sample_count(&self, context: usize, first: bool,
                        action: Action) -> Result<usize, Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        Ok(self.counts[context][usize::from(first)][action.slot()])
    }
    pub fn observations_seen(&self) -> u64 { self.updates }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcomes_are_bounded_and_calibrated() {
        let mut m = ResponseMemory::<2, 4>::new().unwrap();
        assert_eq!(m.predict_readout_one(0, true, Action::Probe), Ok(0.5));
        for _ in 0..6 { m.observe(0,true,Action::Probe,true).unwrap(); }
        assert_eq!(m.sample_count(0,true,Action::Probe), Ok(4));
        assert!((m.predict_readout_one(0,true,Action::Probe).unwrap() - 5.0/6.0).abs() < 1e-12);
        assert_eq!(m.observations_seen(), 6);
        assert_eq!(m.sample_count(0,true,Action::Passive), Ok(0));
    }

    #[test]
    fn computes_branch_value() {
        let got = ResponseMemory::<2,4>::expected_accuracy(0.75,0.60,0.90).unwrap();
        assert!((got - (0.25*0.60+0.75*0.90)).abs()<1e-12);
    }

    #[test]
    fn refuses_invalid_settings() {
        assert!(ResponseMemory::<0,4>::new().is_err());
        assert!(ResponseMemory::<2,0>::new().is_err());
        let m = ResponseMemory::<2,4>::new().unwrap();
        assert_eq!(m.predict_readout_one(2,false,Action::Probe), Err(Error::InvalidContext));
        assert_eq!(ResponseMemory::<2,4>::expected_accuracy(f64::NAN,0.5,0.5), Err(Error::InvalidProbability));
    }
}
