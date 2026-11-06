//! CORVID v0.4: bounded, label-calibrated sensor reliability and online
//! contextual observation benefit. No heap storage and no known sensor error
//! rates. This is a research kernel, NOT a verified biological model.
use crate::Error;

/// Store a bounded history for 3 sensors plus the co-failure of sensors 1&2.
#[derive(Debug, Clone)]
pub struct FeedbackMemory<const CONTEXTS: usize, const WINDOW: usize> {
    history: [[[u8; WINDOW]; 4]; CONTEXTS],
    count: [[usize; 4]; CONTEXTS],
    cursor: [[usize; 4]; CONTEXTS],
}

impl<const CONTEXTS: usize, const WINDOW: usize> FeedbackMemory<CONTEXTS, WINDOW> {
    pub fn new() -> Result<Self, Error> {
        if CONTEXTS == 0 || WINDOW == 0 {
            return Err(Error::InvalidProbability);
        }
        Ok(Self {
            history: [[[0u8; WINDOW]; 4]; CONTEXTS],
            count: [[0usize; 4]; CONTEXTS],
            cursor: [[0usize; 4]; CONTEXTS],
        })
    }

    fn append(&mut self, context: usize, channel: usize, error: bool) {
        let index = self.cursor[context][channel];
        self.history[context][channel][index] = u8::from(error);
        self.cursor[context][channel] = (index + 1) % WINDOW;
        self.count[context][channel] = (self.count[context][channel] + 1).min(WINDOW);
    }

    /// Call ONLY AFTER the ground-truth target has been verified.
    pub fn learn(
        &mut self,
        context: usize,
        primary: bool,
        extra: Option<(bool, bool)>,
        verified_truth: bool,
    ) -> Result<(), Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        self.append(context, 0, primary != verified_truth);
        if let Some((secondary, tertiary)) = extra {
            self.append(context, 1, secondary != verified_truth);
            self.append(context, 2, tertiary != verified_truth);
            self.append(context, 3,
                secondary != verified_truth && tertiary != verified_truth);
        }
        Ok(())
    }

    pub fn sample_count(&self, context: usize, sensor: usize) -> Result<usize, Error> {
        if context >= CONTEXTS || sensor >= 4 { return Err(Error::InvalidContext); }
        Ok(self.count[context][sensor])
    }

    /// Beta(1,3) smoothing. A fixed prior is NOT a known error-rate label.
    pub fn estimated_error(&self, context: usize, sensor: usize) -> Result<f64, Error> {
        if context >= CONTEXTS || sensor >= 3 { return Err(Error::InvalidContext); }
        let count = self.count[context][sensor];
        let failures: usize = self.history[context][sensor][..count]
            .iter().map(|&e| usize::from(e)).sum();
        Ok((1.0 + failures as f64) / (4.0 + count as f64))
    }

    /// Posterior-smoothed rate of both optional sensors being wrong together.
    pub fn estimated_cofailure(&self, context: usize) -> Result<f64, Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        let count = self.count[context][3];
        let failures: usize = self.history[context][3][..count]
            .iter().map(|&e| usize::from(e)).sum();
        Ok((1.0 + failures as f64) / (8.0 + count as f64))
    }
}

/// A bounded contextual query-value learner. The delayed verified feedback
/// exposes correctness of the baseline and the queried decision on trials
/// where the query was made. It never uses hidden future labels or error rates.
#[derive(Debug, Clone)]
pub struct QueryValue<const CONTEXTS: usize> {
    gain: [f64; CONTEXTS],
    effective_count: [f64; CONTEXTS],
    decay: f64,
}

impl<const CONTEXTS: usize> QueryValue<CONTEXTS> {
    pub fn new(decay: f64) -> Result<Self, Error> {
        if CONTEXTS == 0 || !decay.is_finite() || !(0.0..1.0).contains(&decay) {
            return Err(Error::InvalidProbability);
        }
        Ok(Self { gain: [0.0; CONTEXTS],
                  effective_count: [0.0; CONTEXTS], decay })
    }

    /// Called after trial outcome is observed; both correctness values must
    /// reflect decisions made BEFORE verified ground truth was available.
    pub fn learn(
        &mut self,
        context: usize,
        baseline_correct: bool,
        queried_correct: bool,
    ) -> Result<(), Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        self.gain[context] = self.decay*self.gain[context]
            + (i32::from(queried_correct) - i32::from(baseline_correct)) as f64;
        self.effective_count[context] = self.decay*self.effective_count[context] + 1.0;
        Ok(())
    }

    pub fn estimate(&self, context: usize) -> Result<f64, Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        Ok(self.gain[context] / (self.effective_count[context] + 3.0))
    }

    pub fn ucb_weight(&self, context: usize) -> Result<f64, Error> {
        let gain = self.estimate(context)?;
        let count = self.effective_count[context];
        Ok((1.0 + 5.0*(gain + 0.18/(1.0+count).sqrt())).max(0.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedback_memory_is_bounded_and_calibrates() {
        let mut memory = FeedbackMemory::<2, 8>::new().unwrap();
        assert!((memory.estimated_error(0, 0).unwrap() - 0.25).abs() < 1e-9);
        for _ in 0..16 { memory.learn(0, true, Some((false,false)), false).unwrap(); }
        assert_eq!(memory.sample_count(0, 0).unwrap(), 8);
        assert!(memory.estimated_error(0, 0).unwrap() > 0.7);
        assert!(memory.estimated_error(0, 1).unwrap() < 0.15);
        assert!((memory.estimated_error(1, 0).unwrap() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn correlated_failure_is_detected_from_delayed_feedback() {
        let mut m = FeedbackMemory::<1, 16>::new().unwrap();
        for _ in 0..16 { m.learn(0, false, Some((true,true)), false).unwrap(); }
        assert!(m.estimated_cofailure(0).unwrap() > 0.7);
        for _ in 0..16 { m.learn(0, false, Some((false,false)), false).unwrap(); }
        assert!(m.estimated_cofailure(0).unwrap() < 0.1);
        assert_eq!(m.learn(1, false, None, false), Err(Error::InvalidContext));
    }

    #[test]
    fn query_value_changes_direction_from_feedback() {
        let mut q = QueryValue::<2>::new(0.95).unwrap();
        for _ in 0..20 {
            q.learn(0, false, true).unwrap();
            q.learn(1, true, false).unwrap();
        }
        assert!(q.estimate(0).unwrap() > 0.5);
        assert!(q.estimate(1).unwrap() < -0.5);
        assert!(q.ucb_weight(0).unwrap() > q.ucb_weight(1).unwrap());
        assert!(QueryValue::<2>::new(1.1).is_err());
    }
}
