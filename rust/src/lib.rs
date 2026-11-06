#![forbid(unsafe_code)]
//! METARGON research kernel (historical components developed as CORVID 0.2–0.7).
//!
//! This is an experimental cognitive kernel, not a neural model, and does
//! not establish biological fidelity or novelty. All inference state is
//! bounded by a compile-time context count; no heap allocation is required.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidContext,
    InvalidProbability,
}

#[derive(Debug, Clone, Copy)]
pub struct SensorReading {
    pub value: bool,
    /// Probability that the sensor's reported binary value is wrong.
    pub error_rate: f64,
}

impl SensorReading {
    pub fn new(value: bool, error_rate: f64) -> Result<Self, Error> {
        if !error_rate.is_finite() || !(0.0..0.5).contains(&error_rate) {
            return Err(Error::InvalidProbability);
        }
        Ok(Self { value, error_rate })
    }
}

fn p_observation(reading: SensorReading, true_value: bool) -> f64 {
    if reading.value == true_value {
        1.0 - reading.error_rate
    } else {
        reading.error_rate
    }
}

/// Compute P(hidden cue = 1 | readings) with an even cue prior.
/// The optional second reading is assumed conditionally independent.
pub fn cue_posterior(first: SensorReading, second: Option<SensorReading>) -> f64 {
    let mut likelihood0 = p_observation(first, false);
    let mut likelihood1 = p_observation(first, true);
    if let Some(extra) = second {
        likelihood0 *= p_observation(extra, false);
        likelihood1 *= p_observation(extra, true);
    }
    let normalizer = likelihood0 + likelihood1;
    if normalizer == 0.0 { 0.5 } else { likelihood1 / normalizer }
}

fn reward_probability(mode: bool, action: bool, p_cue1: f64, error: f64) -> f64 {
    // The (hidden) correct action is cue XOR mode.
    let correct_when_zero = action == mode;
    let correct_when_one = action != mode;
    let p_correct = if correct_when_zero { 1.0 - p_cue1 } else { 0.0 }
        + if correct_when_one { p_cue1 } else { 0.0 };
    error + (1.0 - 2.0 * error) * p_correct
}

#[derive(Debug, Clone, Copy)]
pub struct Decision {
    context: usize,
    p_cue1: f64,
    pub action: bool,
    /// Posterior P(mode=1) at the time of action selection.
    pub mode_belief: f64,
}

#[derive(Debug, Clone)]
pub struct Core<const CONTEXTS: usize> {
    mode_probabilities: [f64; CONTEXTS],
    transition_hazard: f64,
    reward_error: f64,
    pub reasoning_updates: u64,
}

impl<const CONTEXTS: usize> Core<CONTEXTS> {
    /// transition_hazard is mode-switch probability per trial, not per idle tick.
    pub fn new(transition_hazard: f64, reward_error: f64) -> Result<Self, Error> {
        if !transition_hazard.is_finite()
            || !reward_error.is_finite()
            || !(0.0..0.5).contains(&transition_hazard)
            || !(0.0..0.5).contains(&reward_error)
        {
            return Err(Error::InvalidProbability);
        }
        Ok(Self {
            mode_probabilities: [0.5; CONTEXTS],
            transition_hazard,
            reward_error,
            reasoning_updates: 0,
        })
    }

    pub fn mode_belief(&self, context: usize) -> Result<f64, Error> {
        self.mode_probabilities
            .get(context)
            .copied()
            .ok_or(Error::InvalidContext)
    }

    /// Call once per trial after receiving a cue (not for irrelevant ticks).
    pub fn decide(
        &mut self,
        context: usize,
        first: SensorReading,
        second: Option<SensorReading>,
    ) -> Result<Decision, Error> {
        let prior = self.mode_probabilities
            .get_mut(context)
            .ok_or(Error::InvalidContext)?;
        *prior = *prior * (1.0 - self.transition_hazard)
            + (1.0 - *prior) * self.transition_hazard;
        let belief = *prior;
        let p_cue1 = cue_posterior(first, second);
        let score_false = (1.0 - belief) * reward_probability(false, false, p_cue1, self.reward_error)
            + belief * reward_probability(true, false, p_cue1, self.reward_error);
        let score_true = (1.0 - belief) * reward_probability(false, true, p_cue1, self.reward_error)
            + belief * reward_probability(true, true, p_cue1, self.reward_error);
        self.reasoning_updates += 1;
        Ok(Decision {
            context,
            p_cue1,
            action: score_true > score_false,
            mode_belief: belief,
        })
    }

    /// Update the mode belief from an observed binary outcome.
    /// Reward noise is modeled; hidden mode is never revealed to the agent.
    pub fn learn(&mut self, decision: Decision, reward: bool) -> Result<(), Error> {
        let prior = self.mode_probabilities
            .get_mut(decision.context)
            .ok_or(Error::InvalidContext)?;
        let prob0 = reward_probability(false, decision.action, decision.p_cue1, self.reward_error);
        let prob1 = reward_probability(true, decision.action, decision.p_cue1, self.reward_error);
        let l0 = if reward { prob0 } else { 1.0 - prob0 };
        let l1 = if reward { prob1 } else { 1.0 - prob1 };
        *prior = *prior * l1 / ((1.0 - *prior) * l0 + *prior * l1);
        self.reasoning_updates += 1;
        Ok(())
    }
}

/// A finite online budget for optional observations.
#[derive(Debug, Clone, Copy)]
pub struct ObservationBudget {
    remaining: usize,
}

impl ObservationBudget {
    pub fn new(allowance: usize) -> Self { Self { remaining: allowance } }
    pub fn remaining(&self) -> usize { self.remaining }

    /// A simple illustrative rule: spend a token only when quality is poor.
    /// The caller may implement a learned policy later.
    pub fn request(&mut self, confidence_is_low: bool) -> bool {
        if !confidence_is_low || self.remaining == 0 { return false; }
        self.remaining -= 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_respects_sensor_quality() {
        let noisy = SensorReading::new(true, 0.42).unwrap();
        let reliable = SensorReading::new(false, 0.08).unwrap();
        assert!((cue_posterior(noisy, None) - 0.58).abs() < 1e-9);
        assert!(cue_posterior(noisy, Some(reliable)) < 0.5);
    }

    #[test]
    fn bounded_contexts_are_independent() {
        let mut core = Core::<2>::new(0.012, 0.05).unwrap();
        let cue = SensorReading::new(false, 0.03).unwrap();
        for _ in 0..12 {
            let decision = core.decide(0, cue, None).unwrap();
            // Hidden mode=0, cue=0, so action=false is the correct one.
            core.learn(decision, !decision.action).unwrap();
        }
        assert!(core.mode_belief(0).unwrap() < 0.1);
        assert_eq!(core.mode_belief(1).unwrap(), 0.5);
        assert_eq!(core.reasoning_updates, 24);
    }

    #[test]
    fn invalid_observation_is_rejected() {
        assert_eq!(SensorReading::new(false, f64::NAN).unwrap_err(), Error::InvalidProbability);
        assert_eq!(SensorReading::new(false, 0.5).unwrap_err(), Error::InvalidProbability);
        assert!(Core::<4>::new(0.01, -0.1).is_err());
        let mut core = Core::<4>::new(0.012, 0.05).unwrap();
        assert_eq!(core.decide(8, SensorReading::new(false, 0.03).unwrap(), None)
                   .unwrap_err(), Error::InvalidContext);
    }

    #[test]
    fn observation_budget_is_hard_cap() {
        let mut budget = ObservationBudget::new(2);
        assert!(!budget.request(false));
        assert!(budget.request(true));
        assert!(budget.request(true));
        assert!(!budget.request(true));
        assert_eq!(budget.remaining(), 0);
    }
}

/// CORVID v0.3: infer a primary sensor's error probability using a bounded
/// history of paired sensor readings. No quality label is accepted as input.
///
/// Assumes conditional independence between primary and secondary sensor
/// errors, with a *calibrated/known* secondary sensor error rate. With q =
/// secondary error, P(disagreement) = q + (1 - 2q) * primary_error.
/// This assumption must be validated with physical sensors before deployment.
#[derive(Debug, Clone)]
pub struct OnlineReliability<const CONTEXTS: usize, const WINDOW: usize> {
    disagreements: [[u8; WINDOW]; CONTEXTS],
    counts: [usize; CONTEXTS],
    offsets: [usize; CONTEXTS],
    prior_error: f64,
    secondary_error: f64,
}

impl<const CONTEXTS: usize, const WINDOW: usize> OnlineReliability<CONTEXTS, WINDOW> {
    pub fn new(prior_error: f64, secondary_error: f64) -> Result<Self, Error> {
        if WINDOW == 0 || CONTEXTS == 0 || !prior_error.is_finite()
            || !secondary_error.is_finite()
            || !(0.0..0.5).contains(&prior_error)
            || !(0.0..0.5).contains(&secondary_error)
        {
            return Err(Error::InvalidProbability);
        }
        Ok(Self {
            disagreements: [[0; WINDOW]; CONTEXTS],
            counts: [0; CONTEXTS],
            offsets: [0; CONTEXTS],
            prior_error,
            secondary_error,
        })
    }

    pub fn observe(
        &mut self,
        context: usize,
        first: bool,
        second: bool,
    ) -> Result<(), Error> {
        let buffer = self.disagreements.get_mut(context).ok_or(Error::InvalidContext)?;
        let next = &mut self.offsets[context];
        buffer[*next] = u8::from(first != second);
        *next = (*next + 1) % WINDOW;
        self.counts[context] = (self.counts[context] + 1).min(WINDOW);
        Ok(())
    }

    pub fn sample_count(&self, context: usize) -> Result<usize, Error> {
        self.counts.get(context).copied().ok_or(Error::InvalidContext)
    }

    pub fn estimate_error(&self, context: usize) -> Result<f64, Error> {
        let buffer = self.disagreements.get(context).ok_or(Error::InvalidContext)?;
        let count = self.counts[context];
        let disagreement_sum: usize = buffer.iter().take(count).map(|value| *value as usize).sum();
        let q = self.secondary_error;
        let prior_disagreement = q + (1.0 - 2.0*q) * self.prior_error;
        const SHRINKAGE: f64 = 2.0;
        let disagreement_rate = (disagreement_sum as f64 + SHRINKAGE*prior_disagreement)
            / (count as f64 + SHRINKAGE);
        Ok(((disagreement_rate - q) / (1.0 - 2.0*q)).clamp(0.01, 0.49))
    }
}

#[cfg(test)]
mod online_reliability_tests {
    use super::*;

    #[test]
    fn learns_and_forgets_without_quality_labels() {
        let mut c = OnlineReliability::<2, 16>::new(0.225, 0.08).unwrap();
        assert!((c.estimate_error(0).unwrap()-0.225).abs() < 1e-12);
        for _ in 0..16 { c.observe(0, false, true).unwrap(); }
        assert!(c.estimate_error(0).unwrap() > 0.40);
        for _ in 0..16 { c.observe(0, false, false).unwrap(); }
        assert!(c.estimate_error(0).unwrap() < 0.1);
        assert_eq!(c.sample_count(0).unwrap(), 16);
        assert_eq!(c.sample_count(1).unwrap(), 0);
    }

    #[test]
    fn rejecting_invalid_parameters_and_contexts() {
        assert!(OnlineReliability::<2, 0>::new(0.2, 0.08).is_err());
        assert!(OnlineReliability::<0, 16>::new(0.2, 0.08).is_err());
        assert!(OnlineReliability::<2, 16>::new(f64::NAN, 0.08).is_err());
        assert!(OnlineReliability::<2, 16>::new(0.2, 0.5).is_err());
        let mut c = OnlineReliability::<2, 8>::new(0.2, 0.08).unwrap();
        assert_eq!(c.observe(8, false, true), Err(Error::InvalidContext));
    }
}

/// CORVID v0.4 supervised, bounded-memory attention components.
pub mod v04;

/// Sparse, delayed verified-outcome learning research kernel (v0.5).
pub mod v05;

/// Bounded verified-feedback memory for passive versus active diagnostics.
pub mod v06;

/// Bounded one-step action response predictor (v0.7 research kernel).
pub mod v07;

/// New v0.8 bounded gate-transition belief tracker.
pub mod v08;
