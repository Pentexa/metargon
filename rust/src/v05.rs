//! CORVID v0.5. Fixed-size, sparse verified-feedback memory.
//!
//! In deployment the trusted environment must provide labels only after a
//! separately accounted and approved request, subject to a delivery delay.
//! This module NEVER estimates accuracy from agreement among unlabeled
//! sensors. It has no allocation and stores only bounded feedback records.
use crate::Error;

/// Fixed-size per-context error observations for the three sensors.
#[derive(Debug, Clone)]
pub struct SparseFeedback<const CONTEXTS: usize, const WINDOW: usize> {
    values: [[[u8; WINDOW]; 4]; CONTEXTS],
    len: [[usize; 4]; CONTEXTS],
    cursor: [[usize; 4]; CONTEXTS],
    verified_count: [u64; CONTEXTS],
    decay: f64,
}

impl<const CONTEXTS: usize, const WINDOW: usize> SparseFeedback<CONTEXTS, WINDOW> {
    pub fn new(decay: f64) -> Result<Self, Error> {
        if CONTEXTS == 0 || WINDOW == 0 || !decay.is_finite() || decay <= 0.0 || decay > 1.0 {
            return Err(Error::InvalidProbability);
        }
        Ok(Self {
            values: [[[0; WINDOW]; 4]; CONTEXTS],
            len: [[0; 4]; CONTEXTS],
            cursor: [[0; 4]; CONTEXTS],
            verified_count: [0; CONTEXTS],
            decay,
        })
    }

    fn append(&mut self, context: usize, sensor: usize, value: bool) {
        let index = self.cursor[context][sensor];
        self.values[context][sensor][index] = u8::from(value);
        self.cursor[context][sensor] = (index + 1) % WINDOW;
        self.len[context][sensor] = (self.len[context][sensor] + 1).min(WINDOW);
    }

    /// Must be called only when an authenticated truth label has been
    /// delivered. Never call based only on the decision's success guess.
    pub fn deliver_verified_label(
        &mut self, context: usize, primary: bool,
        extra: Option<(bool,bool)>, truth: bool,
    ) -> Result<(), Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        self.verified_count[context] = self.verified_count[context].saturating_add(1);
        self.append(context, 0, primary != truth);
        if let Some((b,c)) = extra {
            self.append(context, 1, b != truth);
            self.append(context, 2, c != truth);
            self.append(context, 3, b != truth && c != truth);
        }
        Ok(())
    }

    fn estimate(&self, context: usize, channel: usize, a: f64, b: f64) -> f64 {
        let n = self.len[context][channel];
        let start = (self.cursor[context][channel] + WINDOW - n) % WINDOW;
        let mut errors = 0.0;
        let mut weight_sum = 0.0;
        for i in 0..n {
            let index = (start + i) % WINDOW;
            let weight = self.decay.powi((n - 1 - i) as i32);
            errors += f64::from(self.values[context][channel][index]) * weight;
            weight_sum += weight;
        }
        (a + errors) / (a + b + weight_sum)
    }

    pub fn error_probability(&self, context: usize, sensor: usize) -> Result<f64, Error> {
        if context >= CONTEXTS || sensor >= 3 { return Err(Error::InvalidContext); }
        Ok(self.estimate(context,sensor,1.0,3.0))
    }

    pub fn pair_failure_probability(&self, context: usize) -> Result<f64, Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        Ok(self.estimate(context,3,1.0,7.0))
    }

    pub fn observation_count(&self, context: usize, sensor: usize) -> Result<usize, Error> {
        if context >= CONTEXTS || sensor >= 4 { return Err(Error::InvalidContext); }
        Ok(self.len[context][sensor])
    }

    pub fn verified_count(&self, context: usize) -> Result<u64, Error> {
        self.verified_count.get(context).copied().ok_or(Error::InvalidContext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_labels_are_bounded_and_allow_inversion() {
        let mut m = SparseFeedback::<2,8>::new(0.93).unwrap();
        assert!((m.error_probability(0,0).unwrap() - 0.25).abs() < 1e-8);
        for _ in 0..20 {
            m.deliver_verified_label(0,true,Some((false,false)),false).unwrap();
        }
        assert_eq!(m.observation_count(0,0).unwrap(),8);
        assert_eq!(m.verified_count(0).unwrap(),20);
        assert!(m.error_probability(0,0).unwrap() > 0.6);
        assert!(m.error_probability(0,1).unwrap() < 0.15);
        assert!((m.error_probability(1,0).unwrap() - 0.25).abs() < 1e-8);
        assert_eq!(m.deliver_verified_label(2,false,None,false),Err(Error::InvalidContext));
    }

    #[test]
    fn old_feedback_is_overwritten() {
        let mut m=SparseFeedback::<1,4>::new(1.0).unwrap();
        for _ in 0..4 {m.deliver_verified_label(0,true,None,false).unwrap();}
        assert!(m.error_probability(0,0).unwrap() > 0.5);
        for _ in 0..4 {m.deliver_verified_label(0,false,None,false).unwrap();}
        assert!(m.error_probability(0,0).unwrap() < 0.2);
        assert!(SparseFeedback::<1,4>::new(0.).is_err());
    }
}
