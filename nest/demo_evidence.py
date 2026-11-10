"""METARGON NEST — automated demonstration evidence.

Runs the six mandatory demos headlessly against the REAL Academy agent and
writes a transcript. Used as the demonstration evidence delivered with the
NEST branch.

Usage:
    python nest/demo_evidence.py [-o nest/demo_evidence.txt]
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
import sys
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from nest.server import NestApp
from nest.agent_session import Challenge


def _run_episode(session, budget=80, timeout=90):
    steps = 0
    deadline = time.time() + timeout
    while steps < budget and time.time() < deadline:
        if session.env is not None and (session.env.finished
                                        or session.env.steps >= session.env.max_steps):
            break
        steps += session.run_steps(1)
        if session.env is not None and (session.env.finished
                                        or session.env.steps >= session.env.max_steps):
            break
    return steps


def _ascii(text: str) -> str:
    """Map common Unicode punctuation to ASCII so the evidence file reads
    cleanly on any terminal/editor (file stays UTF-8)."""
    table = {"—": "-", "–": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
             "…": "...", "→": "->"}
    for src, dst in table.items():
        text = text.replace(src, dst)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default=str(_ROOT / "nest" / "demo_evidence.txt"))
    args = ap.parse_args(argv)

    lines = []
    say = lambda s="": lines.append(_ascii(s))

    with tempfile.TemporaryDirectory() as td:
        app = NestApp(storage_root=Path(td), api_adapter=None)
        s = app.session

        say("METARGON NEST — automated demonstration evidence (runs headless)")
        say("Mode: no API key; deterministic interpreter; real Academy agent + environment")
        say("=" * 72)

        # ---- Demo 1: first encounter ----
        say("\n[Demo 1] First encounter")
        r = app.handle_command("Hello, Metargon!")
        say(f"  user:  Hello, Metargon!")
        say(f"  intent: {r['command']['intent']}")
        say(f"  metargon: {r['reply']}")
        r = app.handle_command("What do you remember?")
        say(f"  user:  What do you remember?")
        say(f"  intent: {r['command']['intent']}")
        say(f"  metargon: {r['reply']}")

        # ---- Demo 2: first task ----
        say("\n[Demo 2] First task: open a door")
        r = app.handle_command("Can you open that door?")
        say(f"  user:  Can you open that door?")
        say(f"  intent: {r['command']['intent']} (focus door: D1)")
        say(f"  metargon: {r['reply']}")
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy", seed=0,
                                  focus_door="D1", description="open door D1"))
        n = _run_episode(s, budget=80)
        w = s.world_snapshot()
        opened = [d for d, o in w["doors"].items() if o]
        say(f"  ran {n} real steps; last action: {w['last_action']}; "
            f"doors open: {opened}; episode {'finished' if w['finished'] else 'unresolved'}")
        say(f"  trace: plans_used={s.agent.trace['plans_used']} "
            f"random_fallback={s.agent.trace['random_fallback_steps']} "
            f"memory_actions={s.agent.trace['memory_actions']}")

        # ---- Demo 3: learning ----
        say("\n[Demo 3] Learning")
        before = s.memory_snapshot()["generated"]
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy", seed=0,
                                  description="train"))
        n = _run_episode(s, budget=80)
        after = s.memory_snapshot()
        say(f"  ran a fresh training episode ({n} steps)")
        say(f"  hypotheses generated: {before} -> {after['generated']}")
        say(f"  status counts now: {after['stats']}")
        sample = after["hypotheses"][:3]
        for h in sample:
            say(f"  hypothesis: door {h['door']} when {h['antecedent_text']} "
                f"[{h['status']}, sup {h['sup']}, con {h['con']}]")

        # ---- Demo 4: persistence ----
        say("\n[Demo 4] Persistence")
        mem1 = s.memory_snapshot()
        info = s.save()
        say(f"  saved checkpoint to {info['agent_state']}")
        fresh = app.__class__(storage_root=Path(td), api_adapter=None).session
        meta = fresh.restore()
        mem2 = fresh.memory_snapshot()
        same = (mem1["generated"] == mem2["generated"]
                and len(mem1["hypotheses"]) == len(mem2["hypotheses"]))
        say(f"  restore ok={meta is not None}; memory identical={same} "
            f"(generated {mem1['generated']} vs {mem2['generated']}, "
            f"hypotheses {len(mem1['hypotheses'])} vs {len(mem2['hypotheses'])})")

        # ---- Demo 5: honest failure ----
        say("\n[Demo 5] Honest failure (unsupported tasks)")
        for text in ("Fly to the moon", "Execute /bin/sh", "Give me emotions"):
            r = app.handle_command(text)
            say(f"  user:  {text}")
            say(f"  intent: {r['command']['intent']}")
            say(f"  metargon: {r['reply']}")

        # ---- Demo 6: no API ----
        say("\n[Demo 6] No API — full workflow")
        say(f"  config api_mode = {app.config()['api_mode']} (expected 'none')")
        r = app.handle_command("Open door D2")
        say(f"  user:  Open door D2 -> intent {r['command']['intent']}")
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy", seed=0,
                                  focus_door="D2", description="open door D2"))
        n = _run_episode(s, budget=40)
        say(f"  ran {n} steps; last action {s.last_action}; status {s.status}")
        app.control("save")
        r = app.handle_command("What do you remember?")
        say(f"  user:  What do you remember?")
        say(f"  metargon: {r['reply']}")

        say("\n" + "=" * 72)
        say("End of evidence transcript.")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"evidence written to {out}")


if __name__ == "__main__":
    main()
