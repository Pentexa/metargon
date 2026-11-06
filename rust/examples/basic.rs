use metargon_core::{Core, ObservationBudget, SensorReading};

fn main() {
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
}
