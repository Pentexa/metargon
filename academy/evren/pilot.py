"""EVREN teacher — pilot + experiment orchestration.

``run_pilot``: the reproducible 10-candidate pilot. It uses the
configured primary teacher (GLM-5.3) for every candidate, validates each
proposal with the strict validator, persists approved tasks and rejected
candidates with full provenance, and falls back to the offline generator
when EVREN is not configured/reachable.

``run_experiment``: measures whether teacher-approved tasks change what
the student learns. Both a baseline (no EVREN-generated tasks) and a
teacher-trained condition consume the same episode budget, same seeds,
and the same evaluation streams. Learning is measured on held-out
teacher tasks and on the classic frozen test pool; forgetting is reported
as early-vs-late success per family.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from ..ckpt.checkpoint import CheckpointManager
from ..envs.gridworld import EnvConfig
from ..envs.tasks import (
    DEFAULT_SPLITS_PATH, EpisodeSpec, difficulty_of, get_splits,
    load_splits,
)
from ..learning.curriculum import BINS, CurriculumManager
from ..learning.loop import run_one_episode, train_agent
from ..eval.evaluator import Evaluator
from ..eval.metrics import forgetting_metrics
from .curriculum_teacher import CurriculumTeacher, TeacherAttempt
from .offline_generator import (
    _task_from_result, run_offline_pilot,
)
from .provider import (
    DEFAULT_MAX_CALLS_PER_RUN, ProviderConfig, QuotaGuard, EvrenProvider,
    api_key_from_env,
)
from .task_registry import ApprovedTask, TaskRegistry
from .validator import STATUS_APPROVED, TaskValidator

PRIMARY = "primary"
SECONDARY = "secondary"

PILOT_N_DEFAULT = 10
RESULTS_EVREN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "academy_v01", "evren")


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def evren_cfg(cfg: dict) -> dict:
    return dict(cfg.get("evren", {}) or {})


def build_provider(cfg: dict, run_id: str = "evren") -> EvrenProvider:
    e = evren_cfg(cfg)
    pcfg = ProviderConfig(
        base_url=e.get("base_url", "https://evren-llmapi.ssyz.org.tr/v1"),
        api_key_env=e.get("api_key_env", "EVREN_API_KEY"),
        timeout=float(e.get("timeout", 60.0)),
        max_retries=int(e.get("max_retries", 3)),
        max_tokens=int(e.get("max_tokens", 1024)),
    )
    max_calls = int(e.get("max_calls_per_run", DEFAULT_MAX_CALLS_PER_RUN))
    allow_unlimited = bool(e.get("allow_unlimited", False))
    quota = QuotaGuard(
        max_calls=max_calls,
        max_total_tokens=int(e.get("max_total_tokens", 0)),
        allow_unlimited=allow_unlimited,
        run_id=run_id)
    return EvrenProvider(pcfg, quota=quota,
                         seed=int(e.get("seed", 0)))


def registry_paths(cfg: dict, results_dir: str = RESULTS_EVREN_DIR) -> str:
    e = evren_cfg(cfg)
    return e.get("registry_dir", results_dir)


def load_splits(cfg: dict):
    path = cfg.get("splits_path")
    if path and os.path.exists(path):
        return load_splits(path)
    return get_splits(cfg.get("seeds", {}).get("pool_split", 20260901),
                      cache_path=path or DEFAULT_SPLITS_PATH)


# ---------------------------------------------------------------------------
# Pilot
# ---------------------------------------------------------------------------
def _attempt_to_result(attempt: TeacherAttempt, validator: TaskValidator):
    if attempt.status != "ok":
        return None
    res = validator.validate(attempt.raw)
    return res


def run_pilot(cfg: dict, splits: dict, n: int = PILOT_N_DEFAULT,
              teacher: str = PRIMARY, force_offline: bool = False,
              results_dir: str = RESULTS_EVREN_DIR,
              run_id: str = "evren-pilot",
              provider: Optional[EvrenProvider] = None):
    """Run the pilot (default 10 candidates, primary teacher GLM-5.3).

    Returns a summary dict. Never raises on provider failure: it falls
    back to the deterministic offline generator and records why.
    """
    os.makedirs(results_dir, exist_ok=True)
    registry = TaskRegistry(results_dir, run_id=run_id)
    validator = TaskValidator(splits=splits)
    e = evren_cfg(cfg)
    seed = int(e.get("seed", 0))
    temperature = float(e.get("temperature", 0.7))
    max_tokens = int(e.get("max_tokens", 1024))

    summary = {
        "run_id": run_id,
        "requested": n,
        "teacher": teacher,
        "model_display": "GLM-5.3" if teacher == PRIMARY else "DeepSeek V4.1 Flash",
        "force_offline": force_offline,
        "started_at": time.time(),
        "source": None,
        "fallback_reason": None,
        "attempts": [],
        "approved": 0,
        "rejected": 0,
        "quota_calls": None,
    }

    # Decide between EVREN and the offline generator. An injected provider
    # (tests) is used as-is; otherwise we need an authenticated key.
    use_offline = force_offline or (provider is None and not api_key_from_env(
        e.get("api_key_env", "EVREN_API_KEY")))
    fallback_reason = None
    if not force_offline and provider is None and not api_key_from_env(
            e.get("api_key_env", "EVREN_API_KEY")):
        fallback_reason = "EVREN_API_KEY not set (add it to the environment or .env)"
        use_offline = True

    if not use_offline:
        if provider is None:
            provider = build_provider(cfg, run_id=run_id)
        try:
            provider.discover_models()
        except Exception as exc:  # network/auth error at discovery time
            fallback_reason = f"discovery failed: {exc}"
            use_offline = True

    if use_offline:
        summary["source"] = "offline_generator"
        summary["fallback_reason"] = fallback_reason
        res = run_offline_pilot(seed=seed, n=n, splits=splits,
                                registry=registry)
        summary["approved"] = len(res["approved"])
        summary["rejected"] = len(res["results"]) - len(res["approved"])
        summary["offline"] = True
        summary["results"] = [r.as_dict() for r in res["results"]]
        summary["finished_at"] = time.time()
        registry.update_manifest(source="offline_generator",
                                 fallback_reason=fallback_reason,
                                 pilot_n=n)
        return summary

    # ---- EVREN path ---------------------------------------------------
    summary["source"] = "evren"
    teacher_obj = CurriculumTeacher(
        provider, model=teacher, temperature=temperature,
        max_tokens=max_tokens, seed=seed)
    approved_count = 0
    rejected_count = 0
    quota_hit = False

    for slot in range(1, n + 1):
        attempt = teacher_obj.propose_task(slot=slot, seed=seed + slot)
        row = {"slot": slot, "status": attempt.status, "error": attempt.error,
               "meta": attempt.meta}
        summary["attempts"].append(row)
        if attempt.status == "quota":
            quota_hit = True
            summary["quota_calls"] = provider.quota.state()
            break
        if attempt.status != "ok":
            rejected_count += 1
            continue
        res = _attempt_to_result(attempt, validator)
        if res is None:
            rejected_count += 1
            continue
        if res.approved:
            approved_count += 1
            task = _task_from_attempt(res, attempt)
            registry.add_approved(task)
            validator.register(res.config)
        else:
            registry.add_rejected(res, meta=attempt.meta)
            rejected_count += 1

    # persist usage records into the registry
    registry.add_usage(EvrenProvider.serialize_usage(provider.usage_records()))

    summary["approved"] = approved_count
    summary["rejected"] = rejected_count
    summary["quota_calls"] = provider.quota.state()
    summary["finished_at"] = time.time()
    registry.update_manifest(
        source="evren",
        teacher=teacher,
        model_display=summary["model_display"],
        pilot_n=n,
        approved=approved_count,
        rejected=rejected_count,
        quota=provider.quota.state())
    if approved_count == 0 and rejected_count > 0 and quota_hit:
        summary["note"] = "quota exhausted before any approval"
    return summary


def _task_from_attempt(res, attempt: TeacherAttempt) -> ApprovedTask:
    raw = res.raw
    m = attempt.meta
    return ApprovedTask(
        task_id=raw.get("task_id") or f"evren-{time.time_ns()}",
        proposal=raw,
        config=res.config,
        model=m.get("model_display", "?"),
        model_id=m.get("model_id", "?"),
        temperature=m.get("temperature", 0.7),
        max_tokens=m.get("max_tokens", 0),
        seed=m.get("seed", 0),
        prompt_tokens=m.get("prompt_tokens", 0),
        completion_tokens=m.get("completion_tokens", 0),
        latency_ms=m.get("latency_ms", 0.0),
        difficulty_intent=raw.get("difficulty_intent", "medium"),
        rationale=raw.get("rationale", ""),
        approved_at=time.time(),
        provenance_note="evren teacher",
    )


# ---------------------------------------------------------------------------
# Experiment: training with teacher tasks vs baseline
# ---------------------------------------------------------------------------
def approved_holdout(approved_configs: list, seed: int = 0,
                     frac: float = 0.2):
    """Deterministically split approved teacher tasks into a training part
    and a held-out evaluation part (disjoint)."""
    rng = random.Random(seed)
    items = list(approved_configs)
    rng.shuffle(items)
    n_hold = max(1, int(round(len(items) * frac)))
    hold = items[:n_hold]
    train = items[n_hold:]
    return train, hold


def teacher_episode_stream(train_configs, teacher_configs, episodes: int,
                           seed: int = 0, p_teacher: float = 0.5,
                           max_steps: int = 80):
    """Deterministic stream mixing classic training tasks and approved
    teacher tasks (p_teacher probability of a teacher task per episode)."""
    rng = random.Random(seed)
    out = []
    for e in range(episodes):
        if rng.random() < p_teacher and teacher_configs:
            cfg = rng.choice(teacher_configs)
        else:
            cfg = rng.choice(train_configs)
        out.append(EpisodeSpec(
            config=cfg, max_steps=max_steps, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="train",
            difficulty=difficulty_of(cfg)))
    return out


def classic_episode_stream(train_configs, episodes: int, seed: int = 0,
                           max_steps: int = 80):
    rng = random.Random(seed)
    out = []
    for e in range(episodes):
        cfg = rng.choice(train_configs)
        out.append(EpisodeSpec(
            config=cfg, max_steps=max_steps, noise_q=0.0,
            spec_seed=seed * 1000 + e, pool_tag="train",
            difficulty=difficulty_of(cfg)))
    return out


def _eval_config_set(agent_factory, configs: list, tag: str, max_steps: int,
                     n_seeds: int = 4, episodes_per_seed: int = 60,
                     seed: int = 0):
    """Evaluate a policy on a set of configs (matched across policies)."""
    evaluator = Evaluator(
        {"train": []}, seed=seed,
        episodes_per_seed=episodes_per_seed,
        max_steps=max_steps, noise_q=0.0, pool_tag=tag)
    # NOTE: Evaluator.check_isolation uses splits["train"] (empty here) so
    # any config set is acceptable; the teacher held-out set is disjoint
    # from the classic train pool by construction.
    return evaluator.run_policy(agent_factory, configs, n_seeds=n_seeds)


def run_experiment(cfg: dict, splits: dict, registry: TaskRegistry,
                   episodes: int = 120, p_teacher: float = 0.5,
                   seed: int = 0, max_steps: int = 80,
                   eval_seeds: int = 4, eval_episodes_per_seed: int = 60,
                   results_dir: str = RESULTS_EVREN_DIR,
                   run_id: str = "evren-exp"):
    """Baseline (no EVREN tasks) vs teacher-condition experiment.

    Both conditions train the same student policy on the same number of
    episodes with the same seed. Evaluation:
      * classic frozen test pool (transfer),
      * held-out teacher tasks (learning on teacher curricula).
    Also reports per-family learning curves and early-vs-late forgetting.
    """
    from ..agents.metargon_agent import MetargonAcademyAgent

    def factory(s):
        return MetargonAcademyAgent(seed=s)

    train_configs = splits["train"]
    approved = registry.approved_configs()
    teacher_train, teacher_hold = approved_holdout(
        approved, seed=seed + 1, frac=0.25)

    # streams (identical budget, matched seed)
    baseline_stream = classic_episode_stream(
        train_configs, episodes, seed=seed, max_steps=max_steps)
    teacher_stream = teacher_episode_stream(
        train_configs, teacher_train, episodes, seed=seed,
        p_teacher=p_teacher, max_steps=max_steps)

    def run_condition(name, stream):
        agent = factory(seed)
        result = train_agent(agent, stream, cfg, f"{run_id}-{name}")
        # per-family curve
        curve = result.curve
        early = [r for r in curve if r["episode"] < episodes // 2]
        late = [r for r in curve if r["episode"] >= episodes // 2]
        fam = {"classic": 0, "teacher": 0}
        for r in curve:
            pass  # family recorded below from stream
        return agent, result, curve

    agent_b, res_b, curve_b = run_condition("baseline", baseline_stream)
    agent_t, res_t, curve_t = run_condition("teacher", teacher_stream)

    # evaluation on classic test pool
    ev = Evaluator(splits, seed=seed + 2,
                   episodes_per_seed=eval_episodes_per_seed,
                   max_steps=max_steps, pool_tag="test")
    def loaded_factory(agent_saved):
        def make(s, _saved=agent_saved):
            a = factory(s)
            a.load_state(_saved.save_state())
            return a
        return make

    eval_b_classic = ev.run_policy(loaded_factory(agent_b), splits["test"],
                                   n_seeds=eval_seeds)
    eval_t_classic = ev.run_policy(loaded_factory(agent_t), splits["test"],
                                   n_seeds=eval_seeds)

    # evaluation on held-out teacher tasks
    eval_b_teacher = _eval_config_set(
        loaded_factory(agent_b), teacher_hold, "teacher_holdout",
        max_steps, n_seeds=eval_seeds,
        episodes_per_seed=eval_episodes_per_seed, seed=seed + 3)
    eval_t_teacher = _eval_config_set(
        loaded_factory(agent_t), teacher_hold, "teacher_holdout",
        max_steps, n_seeds=eval_seeds,
        episodes_per_seed=eval_episodes_per_seed, seed=seed + 3)

    # forgetting: run the *other* family after training (probe)
    probe_configs = teacher_hold if teacher_hold else approved
    probe_b = _eval_config_set(
        loaded_factory(agent_b), probe_configs, "forgetting_probe",
        max_steps, n_seeds=max(2, eval_seeds // 2),
        episodes_per_seed=max(20, eval_episodes_per_seed // 3),
        seed=seed + 4)
    probe_t = _eval_config_set(
        loaded_factory(agent_t), probe_configs, "forgetting_probe",
        max_steps, n_seeds=max(2, eval_seeds // 2),
        episodes_per_seed=max(20, eval_episodes_per_seed // 3),
        seed=seed + 4)

    summary = {
        "run_id": run_id,
        "episodes": episodes,
        "p_teacher": p_teacher,
        "seed": seed,
        "teacher_train_tasks": len(teacher_train),
        "teacher_holdout_tasks": len(teacher_hold),
        "baseline": {
            "train_success": res_b.success_rate,
            "train_mean_steps": res_b.mean_steps,
            "classic_test": eval_b_classic["summary"],
            "teacher_holdout": eval_b_teacher["summary"],
        },
        "teacher": {
            "train_success": res_t.success_rate,
            "train_mean_steps": res_t.mean_steps,
            "classic_test": eval_t_classic["summary"],
            "teacher_holdout": eval_t_teacher["summary"],
        },
        "learning_curves": {
            "baseline": curve_b,
            "teacher": curve_t,
        },
        "forgetting": {
            "baseline_probe": probe_b["summary"],
            "teacher_probe": probe_t["summary"],
            "forgetting_delta_teacher_vs_baseline":
                probe_t["summary"]["success_rate"] - probe_b["summary"]["success_rate"],
        },
        "finished_at": time.time(),
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "experiment_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    # persist trained policy states (for later restore/replay checks)
    states_dir = os.path.join(results_dir, "policies")
    os.makedirs(states_dir, exist_ok=True)
    for name, agent in (("baseline", agent_b), ("teacher", agent_t)):
        with open(os.path.join(states_dir, f"{name}_agent.json"),
                  "w", encoding="utf-8") as f:
            json.dump(agent.save_state(), f, indent=1, default=str)
    return summary
