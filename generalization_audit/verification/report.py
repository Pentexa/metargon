"""Result model and rendering for the verifier."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .checks import CheckResult, FAIL, INCONCLUSIVE, PASS


@dataclass
class VerificationReport:
    source: str
    checks: list[CheckResult]

    @property
    def summary(self) -> dict:
        counts = {PASS: 0, FAIL: 0, INCONCLUSIVE: 0}
        for c in self.checks:
            counts[c.status] = counts.get(c.status, 0) + 1
        return counts

    @property
    def overall(self) -> str:
        counts = self.summary
        if counts[FAIL] > 0:
            return FAIL
        if counts[INCONCLUSIVE] > 0:
            return INCONCLUSIVE
        return PASS

    def render_text(self, max_detail: int = 6) -> str:
        lines = [f"Verification report — {self.source}",
                 f"overall: {self.overall}  "
                 f"(PASS={self.summary.get(PASS, 0)}, "
                 f"FAIL={self.summary.get(FAIL, 0)}, "
                 f"INCONCLUSIVE={self.summary.get(INCONCLUSIVE, 0)})",
                 ""]
        for c in self.checks:
            lines.append(f"[{c.status:11s}] {c.name}: {c.message}")
            for d in c.detail[:max_detail]:
                lines.append(f"            - {d}")
            if len(c.detail) > max_detail:
                lines.append(f"            … ({len(c.detail) - max_detail} more)")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "overall": self.overall,
            "summary": self.summary,
            "checks": [c.as_dict() for c in self.checks],
        }
