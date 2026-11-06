//! CORVID v0.6: bounded online reliability for active diagnostics.
//!
//! The engine has no device I/O and never receives hidden simulator metadata.
//! Updates require a separately verified truth label. It stores a finite
//! number of sensor error flags in fixed arrays, with no heap allocation.
use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diagnostic {
    Passive,
    Probe,
}

impl Diagnostic {
    fn channel(self) -> usize {
        match self { Self::Passive => 1, Self::Probe => 2 }
    }
}

#[derive(Debug, Clone)]
pub struct DiagnosticMemory<const CONTEXTS: usize, const WINDOW: usize> {
    values: [[[u8; WINDOW]; 3]; CONTEXTS],
    counts: [[usize; 3]; CONTEXTS],
    cursor: [[usize; 3]; CONTEXTS],
    delivered: u64,
}

impl<const CONTEXTS: usize, const WINDOW: usize> DiagnosticMemory<CONTEXTS, WINDOW> {
    pub fn new() -> Result<Self, Error> {
        if CONTEXTS == 0 || WINDOW == 0 { return Err(Error::InvalidContext); }
        Ok(Self {
            values: [[[0; WINDOW]; 3]; CONTEXTS],
            counts: [[0; 3]; CONTEXTS],
            cursor: [[0; 3]; CONTEXTS],
            delivered: 0,
        })
    }

    fn push(&mut self, context: usize, channel: usize, is_error: bool) {
        let cur = self.cursor[context][channel];
        self.values[context][channel][cur] = u8::from(is_error);
        self.cursor[context][channel] = (cur + 1) % WINDOW;
        self.counts[context][channel] = (self.counts[context][channel] + 1).min(WINDOW);
    }

    /// Caller must provide a verified delayed label, never an agent guess.
    /// No optional diagnostic => ONLY primary statistics are updated.
    pub fn deliver(
        &mut self, context: usize, primary: bool,
        diagnostic: Option<(Diagnostic, bool)>, truth: bool,
    ) -> Result<(), Error> {
        if context >= CONTEXTS { return Err(Error::InvalidContext); }
        self.push(context, 0, primary != truth);
        if let Some((kind, observed)) = diagnostic {
            self.push(context, kind.channel(), observed != truth);
        }
        self.delivered = self.delivered.saturating_add(1);
        Ok(())
    }

    pub fn sample_count(&self, context: usize, channel: usize) -> Result<usize, Error> {
        if context >= CONTEXTS || channel >= 3 { return Err(Error::InvalidContext); }
        Ok(self.counts[context][channel])
    }

    pub fn error_rate(&self, context: usize, channel: usize) -> Result<f64, Error> {
        if context >= CONTEXTS || channel >= 3 { return Err(Error::InvalidContext); }
        let count = self.counts[context][channel];
        let sum: usize = self.values[context][channel]
            .iter().take(count).map(|v| usize::from(*v)).sum();
        Ok((1.0 + sum as f64) / (4.0 + count as f64))
    }

    /// Expected Bayesian 0/1 decision accuracy after observing one binary
    /// diagnostic under the *assumption* of conditional independence.
    /// The assumption is knowingly violated for correlated passive errors.
    pub fn expected_accuracy_after(
        p_truth_one: f64, diagnostic_error: f64,
    ) -> Result<f64, Error> {
        if !p_truth_one.is_finite() || !diagnostic_error.is_finite()
            || !(0.0..=1.0).contains(&p_truth_one)
            || !(0.0..=1.0).contains(&diagnostic_error) {
            return Err(Error::InvalidProbability);
        }
        let p = p_truth_one;
        let e = diagnostic_error;
        Ok((p * (1.0-e)).max((1.0-p)*e)
            + (p*e).max((1.0-p)*(1.0-e)))
    }

    pub fn labels_delivered(&self) -> u64 { self.delivered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_diagnostic_does_not_invent_a_reading() {
        let mut m = DiagnosticMemory::<2, 8>::new().unwrap();
        m.deliver(0, true, None, false).unwrap();
        assert_eq!(m.sample_count(0, 0), Ok(1));
        assert_eq!(m.sample_count(0, 1), Ok(0));
        assert_eq!(m.sample_count(0, 2), Ok(0));
        assert_eq!(m.labels_delivered(), 1);
    }

    #[test]
    fn bounded_window_for_both_actions() {
        let mut m = DiagnosticMemory::<2, 8>::new().unwrap();
        for i in 0..100 {
            let action = if i%2 == 0 { Diagnostic::Passive } else { Diagnostic::Probe };
            m.deliver(0, true, Some((action, false)), true).unwrap();
        }
        assert_eq!(m.sample_count(0, 0), Ok(8));
        assert_eq!(m.sample_count(0, 1), Ok(8));
        assert_eq!(m.sample_count(0, 2), Ok(8));
        assert_eq!(m.error_rate(1, 0), Ok(0.25));
        assert!(m.error_rate(0, 1).unwrap() > 0.70);
    }

    #[test]
    fn cannot_use_out_of_range_contexts() {
        let mut m = DiagnosticMemory::<2, 8>::new().unwrap();
        assert_eq!(m.deliver(2, false, None, true), Err(Error::InvalidContext));
        assert_eq!(m.error_rate(0, 3), Err(Error::InvalidContext));
        assert!(DiagnosticMemory::<0,8>::new().is_err());
        assert!(DiagnosticMemory::<2,0>::new().is_err());
    }

    #[test]
    fn theoretical_expected_accuracy_is_bounded() {
        let a = DiagnosticMemory::<2, 8>::expected_accuracy_after(0.5, 0.1).unwrap();
        assert!((a - 0.9).abs() < 1e-12);
        let b = DiagnosticMemory::<2, 8>::expected_accuracy_after(0.5, 0.5).unwrap();
        assert!((b - 0.5).abs() < 1e-12);
        assert!(DiagnosticMemory::<2, 8>::expected_accuracy_after(-0.1, 0.1).is_err());
        assert!(DiagnosticMemory::<2, 8>::expected_accuracy_after(f64::NAN, 0.1).is_err());
    }
}
