//! METARGON Academy v0.1 — bounded inference kernel.
//!
//! A small, heap-light Rust mirror of the Academy's learning core:
//!
//!   * `Atom` / conjunctive `Rule` / disjunctive (`AnyRule`) evaluation,
//!   * `Hypothesis` records with support/contradiction/status transitions,
//!   * `HypothesisMemory` with the same deterministic eviction score as the
//!     Python implementation (used for parity tests),
//!   * `Candidates` generation over a fixed atom vocabulary (bounded).
//!
//! Every module here is `#![forbid(unsafe_code)]` — no unsafe code is
//! permitted anywhere in this crate.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod atoms;
pub mod memory;
pub mod rules;

pub use atoms::{Atom, AtomKind, Candidates};
pub use memory::{Hypothesis, HypothesisMemory, Status};
pub use rules::{AnyRule, Rule};
