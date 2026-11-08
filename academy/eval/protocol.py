"""Public-observation protocol boundary.

Student policies must never receive the hidden ``EnvConfig``. The runner
boundaries (evaluator, learning loop, evaluation grids) pass a *public*
episode spec to non-privileged policies, and the full spec only to
privileged diagnostic comparators (the oracle).

This makes the separation enforceable instead of conventional: a student
policy that attempts to read ``spec.config`` raises an AttributeError
because the public view exposes no ``config`` attribute.
"""
from __future__ import annotations

from types import SimpleNamespace

# The oracle is the only privileged policy: it legitimately needs the
# hidden config to plan perfectly (diagnostic reference, excluded from
# deployable rankings).
PRIVILEGED_POLICY_NAMES = frozenset({"oracle"})


def is_privileged(agent) -> bool:
    """True for privileged diagnostic comparators (e.g. the oracle)."""
    return getattr(agent, "name", None) in PRIVILEGED_POLICY_NAMES


def public_episode_spec(spec):
    """Expose only the public episode fields; intentionally hide
    ``.config`` (the hidden EnvConfig)."""
    return SimpleNamespace(
        max_steps=spec.max_steps,
        noise_q=spec.noise_q,
        spec_seed=spec.spec_seed,
        pool_tag=getattr(spec, "pool_tag", None),
        difficulty=getattr(spec, "difficulty", None),
    )


def spec_for(agent, spec):
    """Full spec for privileged policies, public view for everyone else."""
    if is_privileged(agent):
        return spec
    return public_episode_spec(spec)
