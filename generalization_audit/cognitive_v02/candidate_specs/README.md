# Candidate arm specs (Cognitive Evaluation v0.2)

Upcoming Cognitive Memory v0.2 and Exploration v0.2 candidate branches can
be measured by this audit without modifying audit code. Drop one JSON
file per candidate here (or call `arms.register_candidate` from the
candidate's own harness):

**Trained arm** (JSON):

```json
{
  "name": "mem_v02_candidate",
  "kind": "trained",
  "kwargs": { "memory_atom_transfer": true }
}
```

**Control** (JSON):

```json
{
  "name": "explore_v02_muted",
  "kind": "transform",
  "base": "trained",
  "transform": "wipe"
}
```

The runner loads these at startup and prints the registered arms. The
candidate's policy code itself must stay on the candidate branch; only
the spec is read here. `kwargs` are forwarded verbatim to
`MetargonAcademyAgent(**kwargs)`.
