"""NEST v0.2 — demonstrated multi-turn conversation transcript.

Uses a grounding-aware fake language layer (offline) to demonstrate the
full bridge deterministically: user text -> interpretation -> validated
goal -> real agent execution -> grounded reply -> save/restore.

Real deployments use a user-configured OpenAI-compatible API; the bridge
and grounding contract are identical.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "python", ROOT / "academy"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from nest.server import NestApp
from nest.llm import LlmBudget
from nest.agent_session import Challenge


class DemoLlm:
    """Grounding-aware scripted language layer (offline demo only)."""

    def __init__(self):
        self.api_key = "demo"
        self.base_url = "offline://demo"
        self.model = "demo-grounded"
        self.budget = LlmBudget()

    @property
    def enabled(self):
        return True

    def chat(self, messages, temperature=0.4):
        grounding = None
        for msg in messages:
            if msg["role"] == "system" and "GROUNDING" in msg["content"]:
                import json as _json
                c = msg["content"]
                grounding = _json.loads(c[c.find("{"):])
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        low = user.lower()
        mem = grounding["memory"] if grounding else {}
        world = grounding["world"] if grounding else {}
        verified = mem.get("verified_count", 0)
        hv = (mem.get("top_hypotheses") or [])[:1]

        if "remember" in low and "learned" in low:
            return "Öğrendiklerimi şimdi kaydedeceğim. [[ACTION:persistence_save]]"
        if "hatırlıyorsun" in low or ("remember" in low and "what" in low):
            if hv and verified:
                h = hv[0]
                return (f"Sana {verified} doğrulanmış kuraldan söz edebilirim. En güçlüsü: "
                        f"kapı {h['door']} açılır {h['antecedent']} — bu doğrulanmış, "
                        f"bu yüzden söyleyebilirim.")
            return "Henüz doğrulanmış kuralım yok, bu yüzden bir kapı kuralı bildiğimi iddia edemem. [[ACTION:none]]"
        if "why" in low or "neden" in low:
            action = world.get("last_action") or "none"
            if world.get("finished"):
                return "Episode başarıyla bitti; son aksiyon hedefe ulaşmamı sağladı. [[ACTION:none]]"
            return (f"Deneme hedefe ulaşmadı. Son aksiyonum {action} idi ve bölüm bitmedi — "
                    f"bu yüzden başarı iddia edemem. [[ACTION:none]]")
        if "open" in low and "door" in low:
            door = "D2" if "d2" in low else "D1"
            return (f"Kapı {door}'yi açmayı deneyeceğim. Henüz başarmadım — üzerinde çalışıyorum. "
                    f"[[ACTION:goal_open_door_{door.lower()}]]")
        if "another" in low or "farklı" in low:
            return "Anlaşıldı — öğrendiklerimi koruyarak farklı bir yaklaşımla yeni bir bölüm deneyeceğim. [[ACTION:challenge_new]]"
        if "remember" in low and "learned" in low:
            return "Öğrendiklerimi şimdi kaydedeceğim. [[ACTION:persistence_save]]"
        if "harder" in low or "zor" in low:
            return "Daha zor bir görev, anlaşıldı. [[ACTION:challenge_harder]]"
        return "Bu, ilginç bir soru. Şu an bildiklerimden yalnızca hafızamdan ve gözlemlerimden konuşabilirim. [[ACTION:none]]"


def _ascii(s):
    for a, b in (("→", "->"), ("—", "-"), ("–", "-"), ("·", "|")):
        s = s.replace(a, b)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output",
                    default=str(ROOT / "nest" / "demo_conversation_v02.txt"))
    args = ap.parse_args(argv)

    lines = []
    say = lambda s="": lines.append(_ascii(s))

    with tempfile.TemporaryDirectory() as td:
        app = NestApp(storage_root=Path(td), llm_client=DemoLlm())
        s = app.session
        say("METARGON NEST v0.2 - genuine conversation demonstration")
        say("language layer: grounding-aware demo model (offline); bridge identical to the API path")
        say("=" * 78)

        say("\n-- Turn 1 (TR): what do you remember? --")
        r = app.handle_command("Metargon, ne hatırlıyorsun?")
        say(f"  founder:  Metargon, ne hatırlıyorsun?")
        say(f"  mode:     {r['mode']}")
        say(f"  metargon: {r['reply']}")

        say("\n-- Turn 2: teach by running a real episode --")
        s.new_challenge(Challenge(pool_tag="train", difficulty="easy", seed=0,
                                  description="train"))
        s.run_steps(80)
        mem = s.memory_snapshot()
        say(f"  ran 80 real steps; verified={mem['stats'].get('verified')} "
            f"generated={mem['generated']}")

        say("\n-- Turn 3 (EN): what do you remember now? --")
        r = app.handle_command("What do you remember?")
        say(f"  founder:  What do you remember?")
        say(f"  metargon: {r['reply']}")

        say("\n-- Turn 4: remember what you learned (save checkpoint) --")
        r = app.handle_command("Remember what you learned.")
        say(f"  founder:  Remember what you learned.")
        say(f"  action:   {r['action']} | note: {r['action_note']}")

        say("\n-- Turn 5: try opening door D1 --")
        r = app.handle_command("Metargon, try opening that door.")
        say(f"  founder:  Metargon, try opening that door.")
        say(f"  action:   {r['action']}")
        say(f"  note:     {r['action_note']}")
        s.run_steps(40)
        w = s.world_snapshot()
        say(f"  reality:  {w['steps']} steps; doors {w['doors']}; last action {w['last_action']}")

        say("\n-- Turn 6: why did that fail? --")
        r = app.handle_command("Why did that fail?")
        say(f"  founder:  Why did that fail?")
        say(f"  metargon: {r['reply']}")

        say("\n-- Turn 7: try another approach --")
        r = app.handle_command("Try another approach.")
        say(f"  founder:  Try another approach.")
        say(f"  action:   {r['action']} | note: {r['action_note']}")
        # continue learning inside the new episode (retained memory)
        s.run_steps(80)
        mem_after = s.memory_snapshot()
        say(f"  reality:  another 80 steps; generated now={mem_after['generated']}")

        say("\n-- Restart with fresh app, same storage, restore --")
        app2 = NestApp(storage_root=Path(td), llm_client=DemoLlm())
        meta = app2.session.restore()
        mem2 = app2.session.memory_snapshot()
        say(f"  restore ok={meta is not None}; memory identical="
            f"{mem['generated'] == mem2['generated']} "
            f"(generated {mem['generated']} vs {mem2['generated']})")
        r = app2.handle_command("What do you remember?")
        say(f"  metargon (after restore): {r['reply']}")

        say("\n" + "=" * 78)
        say("End of v0.2 conversation demonstration.")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"transcript written to {out}")


if __name__ == "__main__":
    main()
