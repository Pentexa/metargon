# METARGON NEST — the interactive home (v0.1)

**Branch:** `feat/metargon-nest-v01`
**Starting commit:** `d001a00d2a05f3e3690e07c2a85c1a7f2b7c81da` (origin/main)

METARGON NEST is a local browser application that gives the founder a first
interactive home for METARGON. It is **not a chatbot demo**: the student
agent (`MetargonAcademyAgent`) and the Academy environment
(`MechanismGrid`) are the real intelligence behind every action.

---

## 1. What it is

Three panels:

| Panel | Shows |
|---|---|
| Conversation | talk to METARGON; commands are interpreted by a deterministic interpreter (no API key needed). Optional user-keyed OpenAI-compatible adapter only translates language → the same structured commands; it never makes decisions. |
| Live World | the actual Academy grid: agent position, doors, switches, plates, objects, goal, current action, outcomes, success/failure. Real `observe → plan → act → learn` loop, with Start / Pause / Continue / Single-step / Stop / Reset. |
| Learning & Memory | actual agent telemetry: hypotheses (verified / contradicted / uncertain), support & contradiction counts, confidence, cross-episode memory flags, trace counters (memory actions, retrievals, planner usage). |

All rendering reads the real `NestSession` snapshot; nothing is fabricated.

## 2. Architecture

```
nest/
  agent_session.py      session controller: real agent + env, worker thread,
                        bounded step loop, telemetry snapshots
  command_interpreter.py deterministic NL → structured command; optional
                        OpenAI-compatible adapter (API only, user key)
  server.py             stdlib ThreadingHTTPServer, localhost by default,
                        JSON API + static UI
  storage.py            application-owned checkpoint dir (default <repo>/.nest/)
  static/index.html      browser UI (vanilla JS, canvas world)
  demo_evidence.py      headless six-demo evidence generator
  tests/                integration tests (the six demos)
  launch.py             Windows launcher
```

- Zero runtime dependencies (Python 3.11 stdlib).
- The agent loop mirrors the Academy experiment harness exactly:
  `obs = env.observe(); action = agent.act(obs); res = env.step(action);
  agent.learn(obs, action, res)` — plus `episode_start(ep, geom)` and
  `finish_episode(success)`.
- The environment's hidden rules (`EnvConfig`) live only in the world
  constructor; they are never handed to the agent. The agent's public
  inputs are limited to `Observation` and `Geom`.

## 3. Conversation → real goals

The deterministic interpreter maps utterances to a closed intent set:

- greetings / help;
- memory questions: "What do you remember?", "Show me your current
  hypothesis", "What did you learn today?" — answered from real memory;
- goals: "Open door D1/D2", "Reach the goal" — a real `Challenge`
  (pool, difficulty, focus door) fed to the session;
- controls: start / pause / continue / single-step / stop / reset;
- persistence: save / restore / fresh start;
- "why did you do that?" — explanation from the last action's telemetry;
- unsupported requests: honest refusal, never fabricated success.

The optional API adapter is strictly a language→intent classifier. If it
is unset or errors, the deterministic interpreter is used; core workflow
never depends on an API key.

## 4. Persistence

- `agent.save_state()`/`load_state()` are used verbatim (checkpoint
  serialization of real memory + RNG + counters).
- Application-owned directory: `METARGON_NEST_HOME` or `<repo>/.nest/`
  (gitignored); no other paths are touched.
- Restore rejects incompatible checkpoints; memory is never invented.
  The UI separates learned memory from conversational history.

## 5. Honesty rules

- "I don't know yet" is allowed; verified claims are limited to what the
  memory's status counts support.
- Unverified hypotheses are presented as hypotheses, not knowledge.
- The agent does not claim emotions, consciousness, or experiences
  outside its telemetry.
- When a task does not succeed, the UI says so (unresolved/ended without
  reaching the goal).

## 6. Safety

- Server binds to `127.0.0.1` by default.
- No arbitrary shell execution; chat never runs system commands.
- Bounded execution: per-episode max steps from the spec plus a global
  4000-step cap; worker is cancellable (Stop), pauses between steps, and
  never blocks the UI.
- No autonomous background jobs beyond the session worker; no hardware
  control; no external services (except optional user-keyed API).

## 7. Launch (Windows)

From the repository root:

```
python nest/launch.py
```

Then open http://127.0.0.1:8765 (auto-opens). Options:
`--port`, `--no-browser`. To enable the optional language adapter, set
`METARGON_NEST_API_KEY` (and optionally `METARGON_NEST_API_URL` /
`METARGON_NEST_MODEL`) before launching. The UI header shows the API mode.

## 8. Tests & automated demo evidence

- `nest/tests/test_nest.py` — 8 integration tests covering the six demos:
  1. first encounter (greeting + honest "no memory yet");
  2. first task (open door D1 → real actions/learning);
  3. learning (hypotheses change / steps advance);
  4. persistence (save → new session → restore → identical memory);
  5. honest failure (unsupported requests are never claimed);
  6. no-API full workflow.
  Plus server endpoint tests.
- `nest/demo_evidence.txt` — committed automated evidence transcript
  (regenerate with `python nest/demo_evidence.py`). Highlights:
  - door D1 opened during the first challenge with 74 real plans used;
  - after an easy training episode: 10 hypotheses generated, **10
    verified**, e.g. `door D1 opens when empty(P1)`;
  - save/restore returned byte-identical memory (10 vs 10 hypotheses);
  - unsupported requests received honest refusals;
  - full workflow ran with `api_mode = none`.

## 9. CI

A lightweight `nest` job runs `python -m unittest discover -s nest/tests`
in CI (≈ seconds; no long benchmarks).

## 10. Definition of Done

The founder can launch NEST, talk to METARGON, assign a supported
challenge, watch its actual decisions, inspect its learning, save
progress, and continue after restart — all driven by the real student
agent, not an LLM impersonation.
