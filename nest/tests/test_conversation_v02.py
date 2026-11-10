"""NEST v0.2 — acceptance tests for genuine multi-turn conversation.

Uses a scripted-but-grounding-aware FakeLlm in place of the network so the
whole bridge is exercised offline: user text -> LLM (grounded) -> validated
action -> real agent execution -> grounded reply -> persistence round trip.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "python", _ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from nest.server import NestApp
from nest.llm import LlmBudget, extract_action_tag, VALID_ACTION_INTENTS
from nest.agent_session import Challenge


class FakeLlm:
    """Mimics an OpenAI-compatible client, grounding replies in the JSON
    grounding block it receives (so assertions prove the bridge passes real
    state to the language layer)."""

    def __init__(self, max_calls=1000):
        self.api_key = "test-key"
        self.base_url = "https://example.invalid/v1"
        self.model = "fake-model"
        self.budget = LlmBudget(max_calls=max_calls)
        self.calls = []

    @property
    def enabled(self):
        return True

    def chat(self, messages, temperature=0.4):
        if self.budget.exhausted:
            raise RuntimeError("API call/cost budget exhausted; fallback mode active")
        self.calls.append(messages)
        grounding = None
        for msg in messages:
            if msg["role"] == "system" and "GROUNDING" in msg["content"]:
                import json as _json
                content = msg["content"]
                brace = content.find("{")
                grounding = _json.loads(content[brace:])
        self.budget.record(100, 50)
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        low = user.lower()
        mem = grounding["memory"] if grounding else {}
        world = grounding["world"] if grounding else {}
        verified = mem.get("verified_count", 0)
        hv = (mem.get("top_hypotheses") or [])[:1]

        if "remember" in low and "learned" in low:
            return "I'll save what I've learned now. [[ACTION:persistence_save]]"
        if "remember" in low and ("what" in low or "learned" in low):
            if hv and verified:
                h = hv[0]
                return (f"I remember {verified} verified rule(s). Strongest: door "
                        f"{h['door']} opens when {h['antecedent']} — that is verified, "
                        f"so I can state it.")
            return "I don't have verified rules yet, so I can't claim I know a door rule. [[ACTION:none]]"
        if "why" in low:
            action = world.get("last_action") or "none"
            finished = world.get("finished")
            if finished:
                return "The episode finished successfully, so the last action contributed to reaching the goal. [[ACTION:none]]"
            return (f"The attempt did not reach the goal. My last action was {action}, "
                    f"and the episode is not finished — so I cannot claim success. [[ACTION:none]]")
        if "open" in low and "door" in low:
            door = "D2" if "d2" in low else "D1"
            return (f"I'll try to open door {door}. I haven't succeeded yet — let me work on it. "
                    f"[[ACTION:goal_open_door_{door.lower()}]]")
        if "another" in low or "different" in low:
            return "Understood — I'll try a different approach with a fresh episode, keeping what I learned. [[ACTION:challenge_new]]"
        if "remember" in low and "learned" in low:
            return "I'll save what I've learned now. [[ACTION:persistence_save]]"
        if "harder" in low:
            return "A harder challenge, understood. [[ACTION:challenge_harder]]"
        return "That's an interesting question. Grounded in what I know right now, I can only speak from my memory and observations. [[ACTION:none]]"


class TestConversationV02(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.llm = FakeLlm()
        self.app = NestApp(storage_root=root, llm_client=self.llm)
        self.session = self.app.session

    def tearDown(self):
        self._tmp.cleanup()

    def test_mode_is_llm_when_client_enabled(self):
        self.assertEqual(self.app.config()["conversation_mode"], "llm")

    def test_fallback_mode_without_client(self):
        app = NestApp(storage_root=Path(self._tmp.name), llm_client=None)
        self.assertEqual(app.config()["conversation_mode"], "command")

    def test_action_tag_extraction_and_validation(self):
        self.assertEqual(extract_action_tag("x [[ACTION:goal_open_door_d1]]"), "goal_open_door_d1")
        self.assertEqual(extract_action_tag("plain reply"), None)
        self.assertIn("goal_open_door_d1", VALID_ACTION_INTENTS)
        self.assertNotIn("evil_system_command", VALID_ACTION_INTENTS)

    def _run_steps(self, n=20):
        self.session.run_steps(n)

    def test_acceptance_dialog_grounded(self):
        # 1. what do you remember (no memory yet -> honest)
        r1 = self.app.handle_command("Metargon, what do you remember?")
        self.assertEqual(r1["mode"], "llm")
        self.assertIn("can't claim", r1["reply"])

        # teach by running an episode (real learning)
        self.session.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                            seed=0, description="train"))
        self._run_steps(80)

        # 2. what do you remember now (grounded in real memory)
        r2 = self.app.handle_command("What do you remember?")
        self.assertIn("verified", r2["reply"])

        # 3. try opening door D1 -> validated goal + execution
        r3 = self.app.handle_command("Try opening door D1.")
        self.assertEqual(r3["action"], "goal_open_door_d1")
        self.assertIn("Goal accepted", r3["action_note"])
        self.assertEqual(self.session.challenge.focus_door, "D1")
        self._run_steps(40)

        # 4. why did that fail (grounded in telemetry, no fake success)
        r4 = self.app.handle_command("Why did that fail?")
        self.assertEqual(r4["mode"], "llm")
        self.assertIn("cannot claim success", r4["reply"])

        # 5. try another approach -> preserved context, new challenge
        r5 = self.app.handle_command("Try another approach.")
        self.assertEqual(r5["action"], "challenge_new")
        self.assertIsNotNone(self.session.challenge)

        # 6. remember what you learned -> save real state
        r6 = self.app.handle_command("Remember what you learned.")
        self.assertEqual(r6["action"], "persistence_save")
        self.assertTrue(self.app.storage.checkpoint_info()["exists"])

    def test_persistence_restore_grounding(self):
        self.session.new_challenge(Challenge(pool_tag="train", difficulty="easy",
                                            seed=0, description="train"))
        self._run_steps(80)
        before = self.session.memory_snapshot()["generated"]
        self.app.handle_command("Remember what you learned.")

        # restart: new app, same storage
        app2 = NestApp(storage_root=Path(self._tmp.name), llm_client=FakeLlm())
        meta = app2.session.restore()
        self.assertIsNotNone(meta)
        after = app2.session.memory_snapshot()["generated"]
        self.assertEqual(before, after)
        r = app2.handle_command("What do you remember?")
        self.assertIn("verified", r["reply"])

    def test_budget_exhaustion_honest_fallback(self):
        llm = FakeLlm(max_calls=1)
        app = NestApp(storage_root=Path(self._tmp.name), llm_client=llm)
        app.handle_command("Hello?")
        self.assertEqual(llm.budget.calls, 1)
        r = app.handle_command("What do you remember?")
        self.assertEqual(r["mode"], "command")
        self.assertIn("language model unavailable", r["reply"])
        self.assertIn("budget exhausted", r["reply"])


if __name__ == "__main__":
    unittest.main()
