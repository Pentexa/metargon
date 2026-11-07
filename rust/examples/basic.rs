use metargon_core::v09::{Atom, CausalMemory, Outcome, Status};
use metargon_core::{Core, ObservationBudget, SensorReading};

fn main() {
    // --- historical v0.2 kernel demo -------------------------------------
    let mut core = Core::<4>::new(0.012, 0.05).expect("valid model configuration");
    let mut budget = ObservationBudget::new(1);

    // A highly uncertain observation; request an independent second reading.
    let initial = SensorReading::new(true, 0.42).unwrap();
    let additional = if budget.request(true) {
        Some(SensorReading::new(false, 0.08).unwrap())
    } else { None };
    let decision = core.decide(0, initial, additional).unwrap();
    println!("action={} prior={:.3}", decision.action, decision.mode_belief);
    core.learn(decision, true).unwrap();
    println!("updated mode belief={:.3}", core.mode_belief(0).unwrap());

    // --- v0.9 compositional causal memory demo ---------------------------
    let mut mem = CausalMemory::new();
    // Context: plate P1 loaded with object A and switch S1 on.
    let ctx = [
        Atom::Load { plate: 1, obj: 0 },
        Atom::Switch { switch: 1, value: true },
    ];
    let outcome = Outcome::Opens { door: 0 };
    for _ in 0..3 {
        mem.observe_event(&ctx, outcome);
        mem.step_advance();
    }
    let q = mem.query(outcome);
    println!("after 3 supporting events: {} usable hypotheses", q.len());
    let verified = q.iter().filter(|h| h.status == Status::Verified).count();
    println!("verified hypotheses: {verified}");
    let (size, evictions, generated, _) = mem.diagnostics();
    println!("memory size={size} generated={generated} evictions={evictions}");
}
