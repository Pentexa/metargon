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

---

# METARGON NEST v0.2 — genuine interactive conversation

## What changed

The conversation layer was redesigned around the real agent:

- **Primary mode: API-based language model** (user-configured,
  OpenAI-compatible). The model receives a sanitized grounding snapshot
  (world state, challenge/status, last action + outcomes, verified and
  unverified hypotheses with confidence/contradictions, telemetry,
  conversational history) and produces grounded replies. It never touches
  hidden state, never invents memories/observations, and can only request
  actions through a validated intent tag (`[[ACTION:<intent>]]`).
- **Offline fallback** remains and is explicitly labeled
  (“command mode”): deterministic interpreter + telemetry-grounded
  replies. It is not presented as conversational intelligence.
- **Multi-turn context**: conversation history is maintained separately
  from causal memory; the model can answer follow-ups (“why?”, “what
  happened next?”, “what did I teach you?”) grounded in real data.
- **Action bridge**: natural-language requests are converted to validated
  goals/questions; execution runs through the existing agent; progress is
  streamed via the live world panel; the UI distinguishes “I will try”
  from “I succeeded”.
- **Budgets**: call/token/cost budgets are tracked; when exhausted or
  when the provider fails, the app degrades honestly to command mode —
  never to fabricated success.

## API configuration (all via environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `METARGON_NEST_API_KEY` | — | user-provided key (never embedded/stored) |
| `METARGON_NEST_API_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `METARGON_NEST_MODEL` | `gpt-4o-mini` | model id |
| `METARGON_NEST_MAX_TOKENS` | 500 | max completion tokens |
| `METARGON_NEST_TIMEOUT` | 15 | request timeout (s) |
| `METARGON_NEST_MAX_CALLS` | 100 | call budget per session |
| `METARGON_NEST_MAX_COST_USD` | 2.0 | approximate cost cap |

The core agent and all offline workflows run without any of these.

## Honesty contract (enforced in the system prompt and the bridge)

- Only grounded data may be cited; verified rules are the only allowed
  causal claims.
- Unverified hypotheses are presented as hypotheses.
- “I don’t know yet” and “the attempt did not finish” are normal.
- Conversation history ≠ causal memory; the two are never conflated.
- No emotions/consciousness claims; no system commands; no hidden-state
  modification.

## Validation

- `nest/tests/test_conversation_v02.py` runs the full acceptance dialog
  with a grounding-aware offline language layer (what do you remember →
  open door D1 → why did it fail → another approach → save → restore →
  grounded answer after restart).
- Existing v0.1 fallback/UI tests continue to pass.
- `nest/demo_conversation_v02.txt` contains a demonstrated multi-turn
  transcript (English + Turkish) showing the real grounding behavior.

---

# Interactive teaching (student-centric)

Teaching never bypasses the student's intelligence. A hint is converted
into an **unverified candidate hypothesis** inserted into the student's
own hypothesis list; `MetargonAcademyAgent.act()` remains the only action
authority.

## How hints interact with the planner and learning loop

1. The user (or the LLM bridge, restricted to interpreting) provides a
   structured hint, e.g. `"try switch S1"` → `{atoms: {sw S1 true}, door:
   D1}`.
2. NEST constructs a `Hypothesis` object with `status="hypothesis"` (i.e.,
   unverified, `sup=0`, `con=0`) and appends it to the student's memory
   list with a teacher marker. No support, reward or hidden information is
   injected; the student's own `_choose_probe()` will evaluate it among its
   candidates, rank it by its own rules, and may accept or reject it.
3. When the student selects that candidate, its own planner
   (`plan_to_atoms`, expansion caps, protection checks) produces the
   actions exactly as for any other hypothesis candidate. NEST never
   executes outside plans.
4. Every action still goes through `env.step()` and every transition
   through `agent.learn()` — so verification/contradiction of the teacher
   claim happens only through real observations. Teacher claims are never
   auto-verified.
5. The teaching panel shows the hint, the experiments performed, the
   hypotheses (formed by the student, including teacher-suggested ones
   with their current status), and metric deltas. Audit data: the
   student's `planned_probes` log marks which hypotheses it chose
   (teacher-suggested hits are counted separately).

## Measured result (honest)

`nest/tests/test_transfer_teaching.py` proves the mechanism:

- Teacher-suggested hypotheses were selected by the student's own `act()`
  (measured probe hits) during guided training — teaching changed behavior.
- On a related unseen task, the trained agent's memory traces
  (`memory_actions`, `hypothesis_retrieved`) are non-zero while the
  memory-wiped control is zero — persistent learning affects later
  decisions.
- The success comparison on this pilot unseen task showed **no positive
  transfer** (fresh 4/4 vs trained 0/4 on the selected config); this is
  reported plainly rather than claimed as a win.

## Controlled learning-validation experiment

`nest/tools/diagnose_learning.py` (and `test_learning_validation.py`)
compares, under matched seeds/budgets and separated training/evaluation:
fresh, trained without guidance, trained with exploration guidance,
trained with an unverified causal claim, trained with misleading guidance,
and trained-then-wiped. Metrics reported per arm: success, mean steps,
invalid actions, verified/contradicted/uncertain/generated counts,
planner/memory traces. A representative run (3 seeds, 4×3 episodes):

| arm | success | mean steps | verified | contradicted | uncertain |
|---|---|---:|---:|---:|---:|
| fresh | 1.00 | 10.0 | 22 | 0 | 8 |
| train_no_guidance | 0.00 | 80.0 | 10 | 10 | 0 |
| train_guidance | 0.00 | 80.0 | 10 | 10 | 0 |
| train_claim | 0.00 | 80.0 | 10 | 10 | 1 |
| train_misleading | 0.00 | 80.0 | 10 | 10 | 0 |
| train_wiped | 0.00 | 80.0 | 10 | 0 | 0 |

Interpretation (honest): on this evaluation task the training phase
itself — with or without teaching — abolished the successful exploratory
behavior that fresh achieves (all trained arms exhaust the step budget and
fail). Teaching did not further degrade beyond training; the claim arm
added one uncertain hypothesis and much higher retrieval. The suspected
mechanism is the same cross-task contradiction observed in Academy v0.6:
trained hypotheses for the training rule dominate exploration on the
unseen rule, and retrieval pressure replaces useful novelty. No positive
transfer is claimed; the infrastructure reports the failure and its trace
evidence.

## Post-training regression — isolated mechanism (diagnosis)

`nest/tools/diagnose_regression.py` (and `test_regression_diagnosis.py`)
reproduces fresh-vs-trained failure on identical evaluation episodes and
isolates the cause with independent mutations (seeds 90000/90037; two
training/eval pairs T1E1, T2E2; identical step budgets).

| arm (T1E1) | success | memory_actions | retrieved |
|---|---:|---:|---:|
| fresh | 1.00 | 4 | 120 |
| trained_full_thr1 | 0.00 | 33 | 370 |
| trained_no_memory_content | 1.00 | 10 | 180 |
| trained_memory_muted (functional) | 1.00 | 6 | 60 |
| trained_explore_state_reset | 0.00 | 33 | 370 |
| trained_counters_reset | 0.00 | 33 | 370 |
| trained_rng_reset | 0.00 | 33 | 370 |
| trained_full_thr2 | 0.00 | 33 | 370 |

Findings:

- **Confirmed mechanism: memory interference.** Both ablations —
  clearing memory content and functionally muting all memory read paths
  (query/bank/transfer/protection) while keeping the same state — restore
  fresh-level success on T1E1. First probe-action divergence from fresh is
  immediate for the trained agent (0) and delayed for the ablated arms
  (6/2), tracing the interference to the trained hypotheses.
- **Negative controls:** resetting exploration state, counters, RNG, or
  raising the transfer threshold to 2 does not change the trained outcome
  — none of these explain the regression.
- **Not universal:** on T2E2 training actually helps (fresh 0.67 →
  trained 1.00). The regression is task-specific: it occurs when the
  trained rule conflicts with the evaluation rule, and the agent keeps
  re-trying its verified but inapplicable hypothesis instead of exploring.
- No NEST-side defect was identified; the observed behavior is the
  student's cross-task interference (a documented Academy characteristic).
  No evaluation-set tuning was performed. Full reproducible report:
  `nest/diagnose_regression_report.json`.


