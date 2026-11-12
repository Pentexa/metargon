"""METARGON NEST v0.2+ — teaching controller (corrected semantics).

Teaching inputs are classified into three kinds and treated differently:

1. EXPLORATION GUIDANCE  ("try switch S1", "test P1")
   -> NOT a causal claim. It only nudges the student's exploration by
   restoring the referenced atom as an untried probe candidate (via its
   own bookkeeping), leaving all action selection to `act()`.
2. UNVERIFIED CAUSAL CLAIM ("I think S1 opens door D1")
   -> Injected as an unverified candidate hypothesis (status "hypothesis",
   sup=0) with an explicit door consequence. It is verified/contradicted
   only through real observations.
3. LEARNING OBJECTIVE ("discover how to open D1")
   -> Sets a goal/focus without any causal atom claim.

Design constraints:
- The student remains the sole action authority; no external planner runs.
- Teacher claims are never auto-verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from academy.agents.memory import HYPOTHESIS, Hypothesis, consequence_key
from academy.envs.gridworld import DOORS, OBJECTS, PLATES, SWITCHES

KIND_GUIDANCE = "guidance"
KIND_CLAIM = "claim"
KIND_OBJECTIVE = "objective"
KIND_NOVELTY = "novelty"

MAX_CLAIMS_PER_DOOR = 1


def _try_atom(atom) -> Optional[tuple]:
    if not isinstance(atom, (tuple, list)) or len(atom) < 2:
        return None
    kind = str(atom[0])
    if kind == "sw" and len(atom) >= 3:
        sw = str(atom[1])
        if sw in SWITCHES:
            return ("sw", sw, bool(atom[2]))
    if kind in ("load", "loadany", "empty") and len(atom) >= 2:
        plate = str(atom[1])
        if plate not in PLATES:
            return None
        if kind == "load" and len(atom) >= 3 and str(atom[2]) in OBJECTS:
            return ("load", plate, str(atom[2]))
        if kind in ("loadany", "empty"):
            return (kind, plate)
    return None


@dataclass
class TeachingInput:
    kind: str
    atoms: frozenset = field(default_factory=frozenset)
    door: Optional[str] = None
    label: str = ""
    source: str = "human"

    def as_dict(self):
        return {"kind": self.kind, "atoms": sorted(self.atoms),
                "door": self.door, "label": self.label,
                "source": self.source}


class TeachingParser:
    """Deterministic EN/TR teaching phrase classifier + validator."""

    CLAIM_MARKERS = ("opens", "açar", "açıyor", "açılır", "opens door")
    GUIDANCE_MARKERS = ("try", "test", "dene", "dene", "araştır",
                        "investigate", "experiment")
    OBJECTIVE_MARKERS = ("discover", "learn how", "öğren", "keşfet",
                         "find out", "bul", "how to open")
    SW_OFF = {"off", "kapalı", "kapat"}
    EMPTY = {"empty", "boşalt", "boş"}
    ANY = {"anything", "herhangi bir", "herhangi"}

    def parse(self, text: str) -> Optional[TeachingInput]:
        low = (" " + text.lower() + " ")
        door = self._find_door(low)
        atoms = []
        for sw in SWITCHES:
            if sw.lower() in low:
                atoms.append(("sw", sw, not any(t in low for t in self.SW_OFF)))
        for p in PLATES:
            if p.lower() not in low:
                continue
            if any(t in low for t in self.EMPTY):
                atoms.append(("empty", p))
            elif any(t in low for t in self.ANY):
                atoms.append(("loadany", p))
            else:
                obj = next((o for o in OBJECTS if o.lower() in low), None)
                atoms.append(("load", p, obj) if obj else ("loadany", p))
        valid = [a for a in atoms if _try_atom(a)]

        # 1) objective: no atom/door-rule claim required
        if any(t in low for t in self.OBJECTIVE_MARKERS):
            return TeachingInput(KIND_OBJECTIVE, door=door, label=text.strip())
        # 2) causal claim: requires atoms + door
        if any(t in low for t in self.CLAIM_MARKERS):
            if valid and door:
                return TeachingInput(KIND_CLAIM, frozenset(valid), door,
                                    label=text.strip())
            if valid and not door:
                # claim about atoms but no door: not fully specified; treat
                # as guidance rather than injecting a partial causal rule
                return TeachingInput(KIND_GUIDANCE, frozenset(valid), door,
                                     label=text.strip())
            return None
        # 3) exploration guidance: atoms (or novelty)
        if valid:
            return TeachingInput(KIND_GUIDANCE, frozenset(valid), door,
                                 label=text.strip())
        if any(t in low for t in ("yeni bir şey", "new", "novel", "farklı")):
            return TeachingInput(KIND_NOVELTY, door=door, label=text.strip())
        return None

    @staticmethod
    def _find_door(text: str) -> Optional[str]:
        for d in DOORS:
            if d.lower() in text:
                return d
        return None


def parse_teaching_text(text: str) -> Optional[TeachingInput]:
    return TeachingParser().parse(text)


def extract_teaching_tags(text: str) -> list[str]:
    import re
    return re.findall(r"\[\[TEACH:([^\]]+)\]\]", text)


def clean_tags(text: str) -> str:
    import re
    return re.sub(r"\[\[(?:ACTION|HINT|TEACH):[^\]]*\]\]", "", text).strip()


class Teaching:
    """Per-session teaching state and logs (corrected semantics)."""

    def __init__(self, session):
        self.session = session
        self.inputs: list[TeachingInput] = []
        self.objectives: list[dict] = []
        self.teacher_ids: set[int] = set()
        self.guidance_atoms: dict[str, set] = {}   # door -> {atom}
        self.experiments: list[dict] = []
        self.stats = {
            "guidance_applied": 0,
            "claims_added": 0,
            "claims_removed": 0,
            "teacher_claim_probes": 0,
            "objectives_added": 0,
            "hypotheses_before": 0, "verified_before": 0,
            "hypotheses_after": 0, "verified_after": 0,
        }

    # -- teaching input ingestion ---------------------------------------
    def apply(self, ti: TeachingInput) -> bool:
        if ti is None:
            return False
        self.inputs.append(ti)
        if ti.kind == KIND_OBJECTIVE:
            self.objectives.append({"text": ti.label,
                                    "t": len(self.experiments)})
            self.stats["objectives_added"] += 1
            if len(self.objectives) > 20:
                del self.objectives[0]
        elif ti.kind == KIND_CLAIM:
            self._inject_claim(ti)
            self.stats["claims_added"] += 1
        elif ti.kind == KIND_GUIDANCE:
            self._apply_guidance(ti)
            self.stats["guidance_applied"] += 1
        elif ti.kind == KIND_NOVELTY:
            self._reset_tried_doors()
            self.stats["guidance_applied"] += 1
        self.session.trace(
            f"Teaching ({ti.kind}) recorded: {ti.label or ti.atoms}")
        return True

    # -- claim injection (unverified candidate hypothesis) --------------
    def _inject_claim(self, ti: TeachingInput):
        agent = self.session.agent
        if agent.geom is None:
            return  # deferred to episode start (see begin_episode_snapshot)
        self._inject_claim_locked(agent, ti)

    def _inject_claim_locked(self, agent, ti: TeachingInput):
        doors = [ti.door] if ti.door else list(agent.geom.door_cell or DOORS)
        for door in doors:
            existing = [h for h in agent.memory.hypotheses
                        if h.h_id in self.teacher_ids and h.door == door]
            if len(existing) >= MAX_CLAIMS_PER_DOOR:
                continue
            h = Hypothesis(
                antecedent=ti.atoms,
                consequence=consequence_key(door, True),
                status=HYPOTHESIS,
                created_step=agent.memory.step,
                last_update=agent.memory.step,
                h_id=agent.memory.next_id,
            )
            agent.memory.next_id += 1
            agent.memory.hypotheses.append(h)
            self.teacher_ids.add(h.h_id)
        agent._collect_probe_log = True

    # -- exploration guidance (no causal claim) -------------------------
    def _apply_guidance(self, ti: TeachingInput):
        """Restore the referenced atom as an untried probe candidate so the
        student can investigate it again. This changes which OPTIONS the
        student sees, never which action it picks."""
        if ti.door:
            doors = [ti.door]
        else:
            doors = list(self.session.agent.geom.door_cell or DOORS) \
                if self.session.agent.geom else list(DOORS)
        for door in doors:
            self.guidance_atoms.setdefault(door, set()).update(ti.atoms)
        self._refresh_tried_for_guidance()

    def _refresh_tried_for_guidance(self):
        agent = self.session.agent
        if agent.geom is None:
            return
        for door, atoms in self.guidance_atoms.items():
            tried = agent.tried.get(door)
            if tried is None:
                continue
            for a in atoms:
                tried.discard(frozenset({a}))

    def _reset_tried_doors(self):
        agent = self.session.agent
        if agent.geom is None:
            return
        for door in agent.geom.door_cell:
            agent.tried[door] = set()

    def clear_all(self):
        self.remove_teacher_hypotheses()
        self.inputs = []
        self.objectives = []
        self.guidance_atoms = {}

    def remove_teacher_hypotheses(self):
        agent = self.session.agent
        before = len(agent.memory.hypotheses)
        agent.memory.hypotheses = [
            h for h in agent.memory.hypotheses if h.h_id not in self.teacher_ids]
        removed = before - len(agent.memory.hypotheses)
        if removed:
            self.stats["claims_removed"] += removed
        self.teacher_ids.clear()

    def has_teacher_claims(self) -> bool:
        return bool(self.teacher_ids)

    def is_active(self) -> bool:
        """Any teaching influence currently present (claims or guidance)."""
        return bool(self.teacher_ids) or bool(self.guidance_atoms)

    def teacher_claim_probe_hits(self) -> int:
        """Count probes chosen inside act() whose source is a teacher claim
        hypothesis (auditable evidence of investigation, not learning)."""
        agent = self.session.agent
        hits = 0
        for door, ante, src in agent.trace.get("planned_probes", []):
            if src != "hypothesis":
                continue
            if any(h.h_id in self.teacher_ids and h.door == door
                   and frozenset(map(tuple, ante)) == h.antecedent
                   for h in agent.memory.hypotheses):
                hits += 1
        return hits

    # -- episode bookkeeping --------------------------------------------
    def begin_episode_snapshot(self):
        agent = self.session.agent
        if agent.geom is not None:
            # flush any pending claims recorded before first episode
            for ti in self.inputs:
                if ti.kind == KIND_CLAIM:
                    self._inject_claim_locked(agent, ti)
            self._refresh_tried_for_guidance()
        mem = self.session.memory_snapshot()
        self.stats["hypotheses_before"] = mem["generated"]
        self.stats["verified_before"] = mem["stats"].get("verified", 0)

    def record_step(self, action: str, result, note: str = ""):
        if result.events:
            for door, ev in result.events:
                self.experiments.append({
                    "door": door, "event": ev, "action": action,
                    "note": note,
                    "step": self.session.global_steps,
                })
                if len(self.experiments) > 200:
                    del self.experiments[0]

    def snapshot_memory_delta(self):
        mem = self.session.memory_snapshot()
        stats = mem["stats"]
        self.stats["hypotheses_after"] = mem["generated"]
        self.stats["verified_after"] = stats.get("verified", 0)
        return {
            "before": {"hypotheses": self.stats["hypotheses_before"],
                       "verified": self.stats["verified_before"]},
            "after": {"hypotheses": self.stats["hypotheses_after"],
                      "verified": self.stats["verified_after"]},
        }

    def as_dict(self):
        self.stats["teacher_claim_probes"] = self.teacher_claim_probe_hits()
        return {
            "semantics": {
                "guidance": "exploration only; never a causal claim",
                "claim": "unverified candidate hypothesis until observed",
                "objective": "goal/focus only",
            },
            "objectives": self.objectives[-10:],
            "inputs": [ti.as_dict() for ti in self.inputs[-8:]],
            "teacher_claims": sorted(self.teacher_ids),
            "guidance_atoms": {d: sorted(a) for d, a in self.guidance_atoms.items()},
            "experiments_tail": self.experiments[-30:],
            "stats": dict(self.stats),
            "memory_delta": self.snapshot_memory_delta(),
            "mechanism": "teacher claims -> unverified candidate hypotheses; "
                         "guidance -> candidate availability; act() is the "
                         "only action authority.",
        }
