"""METARGON Academy — command-line entry points.

Subcommands:
    splits          compute and persist the immutable task splits
    calibrate       measure difficulty bins and build the curriculum schedule
    smoke           tiny end-to-end smoke experiment
    train           train policies on the matched episode stream
    resume          resume a training run from its latest checkpoint
    eval            evaluate trained policies on val/test pools
    ood            evaluate trained policies on the OOD pool
    checkpoints     list stored checkpoints for a run
    verify-restore  checkpoint restore determinism check
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from academy.config import DEFAULT_CONFIG_PATH, DEFAULT_RESULTS_DIR, load_config, merge_overrides  # noqa: E402
from academy.envs.tasks import get_splits, make_episodes  # noqa: E402
from academy.agents.metargon_agent import MetargonAcademyAgent  # noqa: E402
from academy.agents.baselines import (  # noqa: E402
    RandomPolicy, EpisodicPolicy, QLearningPolicy, BayesPlannerPolicy,
    OraclePolicy,
)
from academy.learning.curriculum import BINS  # noqa: E402
from academy.learning.loop import (  # noqa: E402
    calibrate_curriculum, build_schedule, make_train_stream, train_agent,
)
from academy.eval.evaluator import Evaluator, compare_all  # noqa: E402
from academy.ckpt.checkpoint import CheckpointManager  # noqa: E402


def policy_factories(cfg):
    """Return policy factories keyed by policy name. Each factory takes a
    seed and returns a fresh policy instance."""
    t = cfg["training"]
    return {
        "metargon_full": lambda s: MetargonAcademyAgent(
            seed=s, replay_enabled=t.get("replay_on", True),
            max_hypotheses=t.get("memory_max_hypotheses", 128),
            verify_support=t.get("memory_verify_support", 3),
            max_contradict=t.get("memory_max_contradict", 2),
            predict_threshold=t.get("memory_predict_threshold", 1),
            contradictict_window=t.get("memory_contradict_window", 3),
            candidates_per_event=t.get("memory_candidates_per_event", 10),
            pair_base=t.get("memory_pair_base", 5),
            planner_max_expand=t.get("planner_max_expand", 1500),
            probe_max_plans=t.get("probe_max_plans", 4),
            probe_timeout=t.get("probe_timeout", 10)),
        "metargon_noreplay": lambda s: MetargonAcademyAgent(
            seed=s, replay_enabled=False,
            max_hypotheses=t.get("memory_max_hypotheses", 128),
            verify_support=t.get("memory_verify_support", 3),
            max_contradict=t.get("memory_max_contradict", 2),
            predict_threshold=t.get("memory_predict_threshold", 1),
            contradictict_window=t.get("memory_contradict_window", 3),
            candidates_per_event=t.get("memory_candidates_per_event", 10),
            pair_base=t.get("memory_pair_base", 5),
            planner_max_expand=t.get("planner_max_expand", 1500),
            probe_max_plans=t.get("probe_max_plans", 4),
            probe_timeout=t.get("probe_timeout", 10)),
        "random": lambda s: RandomPolicy(seed=s),
        "episodic": lambda s: EpisodicPolicy(
            seed=s, max_mem=t.get("episodic_mem_max", 400)),
        "qlearning": lambda s: QLearningPolicy(
            seed=s, alpha=t.get("q_alpha", 0.30),
            gamma=t.get("q_gamma", 0.95),
            eps0=t.get("q_eps0", 0.35),
            eps_min=t.get("q_eps_min", 0.05)),
        "bayes": lambda s: BayesPlannerPolicy(seed=s),
        "oracle": lambda s: OraclePolicy(seed=s),
    }


def state_path(results_dir, policy):
    return os.path.join(results_dir, "policies", f"{policy}.json")


def save_policy_state(agent, results_dir, policy):
    os.makedirs(os.path.dirname(state_path(results_dir, policy)), exist_ok=True)
    with open(state_path(results_dir, policy), "w", encoding="utf-8") as f:
        json.dump(agent.save_state(), f, indent=1, sort_keys=True, default=str)


def load_policy_state(factory, results_dir, policy):
    p = state_path(results_dir, policy)
    if not os.path.exists(p):
        return None
    agent = factory(0)
    with open(p, "r", encoding="utf-8") as f:
        agent.load_state(json.load(f))
    return agent


def cmd_splits(args, cfg):
    splits = get_splits(cfg["seeds"]["pool_split"], force_compute=True)
    counts = splits["counts"]
    print(f"splits computed: train={counts['train']} val={counts['val']} "
          f"test={counts['test']} ood={counts['ood']} "
          f"(solvable={counts['solvable']})")
    return 0


def cmd_calibrate(args, cfg):
    splits = get_splits(cfg["seeds"]["pool_split"])
    n_bins = cfg["training"].get("episodes_per_block", 40)
    calib = calibrate_curriculum(
        splits["train"], episodes_per_bin=n_bins, seed=cfg["seeds"]["run"])
    total = cfg["training"]["total_episodes"]
    schedule = build_schedule(
        calib, total, episodes_per_block=cfg["training"]["episodes_per_block"],
        seed=cfg["seeds"]["run"],
        success_target=cfg["curriculum"]["success_target"],
        harder_epsilon=cfg["curriculum"]["harder_epsilon"])
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(DEFAULT_RESULTS_DIR, "calibration.json"),
              "w", encoding="utf-8") as f:
        json.dump({"calibration": calib, "schedule": schedule}, f, indent=1)
    print("calibration (measured difficulty):")
    for b, st in calib.items():
        print(f"  {b}: success={st['success_rate']:.3f} "
              f"std={st.get('std', 0.0):.3f} n={st['n']}")
    print(f"schedule: {total} episodes built")
    return 0


def _train_one(policy_name, factory, stream, cfg, results_dir, resume=False,
               eval_at=None):
    t = cfg["training"]
    run_id = f"{policy_name}_v01"
    checkpoint_dir = os.path.join(cfg["checkpoint"]["dir"], policy_name)
    agent = factory(cfg["seeds"]["run"])
    start_episode = 0
    curve_prefix = []
    if resume:
        ckpt = CheckpointManager(checkpoint_dir, run_id, cfg, 1)
        snap = ckpt.load_latest()
        if snap and ckpt.verify(snap):
            agent.load_state(snap["agent"])
            start_episode = snap["episode"] + 1
            curve_prefix = snap.get("extra", {}).get("curve", [])
            print(f"  resumed {policy_name} from episode {start_episode}")
    use_ckpt = policy_name.startswith("metargon")
    ckpt = CheckpointManager(checkpoint_dir, run_id, cfg,
                             cfg["checkpoint"]["keep_last"]) if use_ckpt else None
    checkpoint_paths = []

    def on_episode(agent_, ep_i, rec, curve):
        if use_ckpt and (ep_i + 1) % cfg["checkpoint"]["every_episodes"] == 0:
            path = ckpt.save(ep_i, agent_.save_state(), {},
                             {}, {"curve": curve_prefix + curve[-200:]})
            checkpoint_paths.append(path)

    result = train_agent(
        agent, stream[start_episode:], cfg, run_id,
        replay_enabled=t.get("replay_on", True) and policy_name == "metargon_full",
        replay_frequency=t.get("replay_frequency", 8),
        replay_batch=t.get("replay_batch", 64),
        eval_at=eval_at,
        on_episode=on_episode,
        checkpoint_paths=checkpoint_paths)
    result.curve = curve_prefix + result.curve
    save_policy_state(agent, results_dir, policy_name)
    return result, agent


def cmd_train(args, cfg):
    splits = get_splits(cfg["seeds"]["pool_split"])
    calib_path = os.path.join(DEFAULT_RESULTS_DIR, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
        schedule = calib_data["schedule"]
    else:
        print("calibration not found; running calibration now...")
        calib = calibrate_curriculum(
            splits["train"], episodes_per_bin=cfg["training"]["episodes_per_block"],
            seed=cfg["seeds"]["run"])
        schedule = build_schedule(
            calib, cfg["training"]["total_episodes"],
            episodes_per_block=cfg["training"]["episodes_per_block"],
            seed=cfg["seeds"]["run"],
            success_target=cfg["curriculum"]["success_target"],
            harder_epsilon=cfg["curriculum"]["harder_epsilon"])
    stream = make_train_stream(
        splits["train"], schedule, seed=cfg["seeds"]["run"],
        max_steps=cfg["world"]["max_steps"])
    factories = policy_factories(cfg)
    policies = args.policies.split(",") if args.policies else \
        ["metargon_full", "metargon_noreplay", "random", "episodic",
         "qlearning", "bayes", "oracle"]
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    summary = {}
    for name in policies:
        print(f"[train] {name} ({len(stream)} episodes)")
        result, _ = _train_one(name, factories[name], stream, cfg,
                                DEFAULT_RESULTS_DIR, resume=args.resume)
        with open(os.path.join(DEFAULT_RESULTS_DIR, f"train_{name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "policy": name,
                "episodes": result.episodes,
                "wall_s": round(result.wall_s, 3),
                "success_rate": result.success_rate,
                "mean_steps": result.mean_steps,
                "curve": result.curve,
                "replay_count": result.replay_count,
                "checkpoints": result.checkpoint_paths,
            }, f, indent=1)
        summary[name] = {"wall_s": round(result.wall_s, 3),
                         "success_rate": result.success_rate,
                         "mean_steps": result.mean_steps}
        print(f"    success={result.success_rate:.3f} "
              f"mean_steps={result.mean_steps:.1f} "
              f"wall={result.wall_s:.1f}s")
    with open(os.path.join(DEFAULT_RESULTS_DIR, "train_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return 0


def cmd_eval(args, cfg, kind="test"):
    splits = get_splits(cfg["seeds"]["pool_split"])
    pool = splits[kind]
    factories = policy_factories(cfg)
    evaluator = Evaluator(
        splits, seed=cfg["seeds"]["eval"],
        episodes_per_seed=cfg["evaluation"]["episodes_per_seed"],
        max_steps=cfg["world"]["max_steps"],
        noise_q=0.0 if kind != "ood" else 0.0,
        pool_tag=kind)
    n_seeds = args.seeds or cfg["evaluation"]["seeds"]
    policies = args.policies.split(",") if args.policies else None
    names = list(factories) if policies is None else policies
    # load trained states when available (fair, post-training evaluation)
    results = {}
    for name in names:
        factory = factories[name]
        agent = load_policy_state(factory, DEFAULT_RESULTS_DIR, name)
        if agent is not None:
            # wrap into a factory that recreates the loaded agent per seed
            def make_loaded(seed, _agent=agent, _factory=factory):
                a = _factory(seed)
                a.load_state(_agent.save_state())
                return a
            res = evaluator.run_policy(make_loaded, pool, n_seeds=n_seeds)
        else:
            res = evaluator.run_policy(factory, pool, n_seeds=n_seeds)
        results[name] = res
        s = res["summary"]
        print(f"  {name}: success={s['success_rate']:.3f} "
              f"(CI {s['ci'][0]:.3f}-{s['ci'][1]:.3f}) "
              f"mean_steps={s['mean_steps']:.1f} n={s['n']}")
    out = {"kind": kind, "n_seeds": n_seeds, "results": results}
    with open(os.path.join(DEFAULT_RESULTS_DIR, f"eval_{kind}.json"),
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    return 0


def cmd_smoke(args, cfg):
    """Small, runnable smoke experiment."""
    print("=== METARGON Academy smoke experiment ===")
    splits = get_splits(cfg["seeds"]["pool_split"], force_compute=True)
    counts = splits["counts"]
    print(f"pools: train={counts['train']} val={counts['val']} "
          f"test={counts['test']} ood={counts['ood']}")
    # tiny schedule: 8 episodes, easy+medium only
    schedule = (["easy"] * 4) + (["medium"] * 4)
    stream = make_train_stream(
        splits["train"], schedule, seed=cfg["seeds"]["run"],
        max_steps=cfg["world"]["max_steps"])
    factories = policy_factories(cfg)
    results_dir = os.path.join(DEFAULT_RESULTS_DIR, "smoke")
    os.makedirs(results_dir, exist_ok=True)
    for name in ["metargon_full", "random", "qlearning"]:
        result, _ = _train_one(name, factories[name], stream, cfg, results_dir)
        with open(os.path.join(results_dir, f"train_{name}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"policy": name, "episodes": result.episodes,
                       "success_rate": result.success_rate,
                       "curve": result.curve,
                       "wall_s": result.wall_s}, f, indent=1)
        print(f"  {name}: success={result.success_rate:.2f} "
              f"wall={result.wall_s:.2f}s")
    evaluator = Evaluator(splits, seed=cfg["seeds"]["eval"],
                          episodes_per_seed=4, max_steps=cfg["world"]["max_steps"],
                          pool_tag="test")
    for name in ["metargon_full", "random"]:
        factory = factories[name]
        agent = load_policy_state(factory, results_dir, name)
        if agent is None:
            continue
        def make_loaded(seed, _agent=agent, _factory=factory):
            a = _factory(seed)
            a.load_state(_agent.save_state())
            return a
        res = evaluator.run_policy(make_loaded, splits["test"], n_seeds=2)
        s = res["summary"]
        print(f"  eval {name}: success={s['success_rate']:.2f} (n={s['n']})")
    print("smoke OK")
    return 0


def cmd_checkpoints(args, cfg):
    base = os.path.join(cfg["checkpoint"]["dir"], args.policy or "metargon_full")
    if not os.path.isdir(base):
        print("no checkpoints found")
        return 0
    files = sorted(f for f in os.listdir(base) if f.endswith(".json"))
    for f in files:
        print(os.path.join(base, f))
    return 0


def cmd_verify_restore(args, cfg):
    """Restore determinism: load the latest checkpoint, run the next
    episode twice from the same restored state, and compare traces."""
    from academy.learning.loop import run_one_episode
    splits = get_splits(cfg["seeds"]["pool_split"])
    policy = args.policy or "metargon_full"
    checkpoint_dir = os.path.join(cfg["checkpoint"]["dir"], policy)
    run_id = f"{policy}_v01"
    ckpt = CheckpointManager(checkpoint_dir, run_id, cfg, 1)
    snap = ckpt.load_latest()
    if snap is None or not ckpt.verify(snap):
        print("no valid checkpoint found")
        return 1
    factories = policy_factories(cfg)
    factory = factories[policy]
    a1 = factory(0)
    a1.load_state(snap["agent"])
    a2 = factory(0)
    a2.load_state(snap["agent"])
    spec = stream_spec_for(splits, snap)
    r1 = run_one_episode(a1, spec, collect_trace=True)
    r2 = run_one_episode(a2, spec, collect_trace=True)
    same = (r1["success"] == r2["success"] and r1["steps"] == r2["steps"]
            and r1["trace"] == r2["trace"])
    print(f"restore determinism: success={r1['success']} steps={r1['steps']} "
          f"same_trace={same}")
    return 0 if same else 1


def stream_spec_for(splits, snap):
    from academy.envs.tasks import EpisodeSpec, difficulty_of
    cfg0 = splits["train"][0]
    return EpisodeSpec(config=cfg0.config, max_steps=80, noise_q=0.0,
                       spec_seed=int(snap["extra"].get("spec_seed", 7)),
                       pool_tag="train",
                       difficulty=difficulty_of(cfg0.config))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="academy")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("splits")
    sub.add_parser("calibrate")
    p_smoke = sub.add_parser("smoke")
    p_train = sub.add_parser("train")
    p_train.add_argument("--policies", default=None)
    p_train.add_argument("--resume", action="store_true")
    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--kind", choices=["val", "test", "ood"], default="test")
    p_eval.add_argument("--seeds", type=int, default=None)
    p_eval.add_argument("--policies", default=None)
    p_eval.add_argument("--fresh", action="store_true")
    sub.add_parser("ood")
    p_ck = sub.add_parser("checkpoints")
    p_ck.add_argument("--policy", default=None)
    p_vr = sub.add_parser("verify-restore")
    p_vr.add_argument("--policy", default=None)
    parser.add_argument("--override", action="append", default=None,
                        help="flat override key=value, e.g. evaluation.seeds=8")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.override:
        overrides = {}
        for kv in args.override:
            k, _, v = kv.partition("=")
            try:
                v = int(v)
            except ValueError:
                pass
            overrides[k] = v
        cfg = merge_overrides(cfg, overrides)
        cfg["_fingerprint"] = cfg.get("_fingerprint")

    if args.command == "splits":
        return cmd_splits(args, cfg)
    if args.command == "calibrate":
        return cmd_calibrate(args, cfg)
    if args.command == "smoke":
        return cmd_smoke(args, cfg)
    if args.command == "train":
        return cmd_train(args, cfg)
    if args.command == "eval":
        return cmd_eval(args, cfg, kind=args.kind)
    if args.command == "ood":
        return cmd_eval(args, cfg, kind="ood")
    if args.command == "checkpoints":
        return cmd_checkpoints(args, cfg)
    if args.command == "verify-restore":
        return cmd_verify_restore(args, cfg)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
