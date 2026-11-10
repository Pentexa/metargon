"""METARGON NEST v0.2+ — human-guided learning controller (student-centric).

Design constraint (architecture):
- Human hints must influence METARGON's OWN action selection — they must
  never bypass `MetargonAcademyAgent.act()`. We therefore do NOT plan or
  execute actions externally.
- A hint is translated into an UNVERIFIED candidate hypothesis (status =
  "hypothesis") inserted into the student's memory list, i.e., the same
  candidate path the student uses for its own hypotheses. The student's
  planner/probe machinery then freely evaluates, plans and executes it —
  and may also reject it.
- Verification/contradiction only happens through the student's own
  `learn()` cycle on real observations; teacher claims are never
  auto-verified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from academy.agents.memory import HYPOTHESIS, Hypothesis, consequence_key
from academy.envs.gridworld import DOORS, OBJECTS, PLATES, SWITCHES

HINT_NOVELTY = "novelty"


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
class Hint:
    atoms: frozenset = field(default_factory=frozenset)
    door: Optional[str] = None
    kind: str = "atoms"          # "atoms" | "novelty"
    label: str = ""
    source: str = "human"
    status: str = "unverified-preference"

    def as_dict(self):
        return {"atoms": sorted(self.atoms), "door": self.door,
                "kind": self.kind, "label": self.label,
                "source": self.source, "status": self.status}


class HintParser:
    """Deterministic EN/TR teaching-phrase -> Hint (validation whitelist)."""

    SW_OFF = {"off", "kapalı", "kapat"}
    EMPTY = {"empty", "boşalt", "boş"}
    ANY = {"anything", "herhangi bir", "herhangi"}

    def parse(self, text: str, door: Optional[str] = None) -> Optional[Hint]:
        low = (" " + text.lower() + " ")
        atoms = []
        door = self._find_door(low) or door

        for sw in SWITCHES:
            if sw.lower() in low:
                val = not any(t in low for t in self.SW_OFF)
                atoms.append(("sw", sw, val))
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
        if not atoms and any(t in low for t in ("yeni bir şey", "new", "novel", "farklı")):
            return Hint(kind=HINT_NOVELTY, door=door,
                        label=text.strip(), source="human")
        if not atoms:
            if door:
                return Hint(door=door, label=text.strip(), source="human")
            return None
        valid = [a for a in atoms if _try_atom(a)]
        if not valid:
            return None
        return Hint(atoms=frozenset(valid), door=door,
                    label=text.strip(), source="human")

    @staticmethod
    def _find_door(text: str) -> Optional[str]:
        for d in DOORS:
            if d.lower() in text:
                return d
        return None


def parse_hint_text(text: str) -> Optional[Hint]:
    return HintParser().parse(text)


def extract_hint_tags(text: str) -> list[str]:
    import re
    return re.findall(r"\[\[HINT:([^\]]+)\]\]", text)


def clean_tags(text: str) -> str:
    import re
    return re.sub(r"\[\[(?:ACTION|HINT):[^\]]*\]\]", "", text).strip()


class Teaching:
    """Per-session teaching state, hypothesis injection, logs."""

    def __init__(self, session):
        self.session = session
        self.objectives: list[dict] = []
        self.hints: list[Hint] = []
        self.teacher_ids: set[int] = set()
        self._pending: list[Hint] = []
        self.experiments: list[dict] = []
        self.stats = {"suggestions_added": 0, "suggestions_removed": 0,
                      "teacher_hypothesis_probes": 0,
                      "hypotheses_before": 0, "verified_before": 0,
                      "hypotheses_after": 0, "verified_after": 0}

    # -- objectives / hints --------------------------------------------
    def set_objective(self, text: str):
        self.objectives.append({"text": text, "t": len(self.experiments)})
        if len(self.objectives) > 20:
            del self.objectives[0]

    def add_hint(self, hint: Hint) -> bool:
        if hint is None:
            return False
        self.hints.append(hint)
        if hint.kind != HINT_NOVELTY:
            self.stats["suggestions_added"] += 1
            self._flush_pending()
        self.session.trace(
            f"Teaching hint recorded (unverified preference): {hint.label or hint.kind}")
        return True

    def clear_hints(self):
        self.remove_teacher_hypotheses()
        self._pending = []
        self.hints = []
        self.stats["suggestions_removed"] += 1

    def _flush_pending(self):
        agent = self.session.agent
        if agent.geom is None:
            return  # applied at episode start
        hint = self.hints[-1] if self.hints else None
        if hint is None or hint.kind == HINT_NOVELTY:
            return
        doors = [hint.door] if hint.door else list(agent.geom.door_cell or DOORS)
        for door in doors:
            h = Hypothesis(
                antecedent=hint.atoms,
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

    def remove_teacher_hypotheses(self):
        agent = self.session.agent
        before = len(agent.memory.hypotheses)
        agent.memory.hypotheses = [
            h for h in agent.memory.hypotheses if h.h_id not in self.teacher_ids]
        removed = before - len(agent.memory.hypotheses)
        if removed:
            self.stats["suggestions_removed"] += removed
        self.teacher_ids.clear()

    def has_teacher_suggestion(self) -> bool:
        return bool(self.teacher_ids)

    def teacher_probe_hits(self) -> int:
        """Count probes chosen inside act() whose source is a teacher-
        suggested hypothesis (auditable evidence that teaching influenced
        the student's candidate selection)."""
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
        self._flush_pending()
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
        self.stats["teacher_hypothesis_probes"] = self.teacher_probe_hits()
        return {
            "objectives": self.objectives[-10:],
            "hints": [h.as_dict() for h in self.hints[-8:]],
            "teacher_hypotheses": sorted(self.teacher_ids),
            "experiments_tail": self.experiments[-30:],
            "stats": dict(self.stats),
            "memory_delta": self.snapshot_memory_delta(),
            "mechanism": "teacher hints -> unverified candidate hypotheses "
                         "evaluated by the student's own act(); no external "
                         "plan execution.",
        }
