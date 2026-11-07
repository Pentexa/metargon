"""METARGON Academy v0.1 — autonomous learning infrastructure.

A reproducible continual-learning and curriculum-generation system for the
METARGON research programme (corvid-inspired, resource-constrained agent).

Design rules enforced throughout the package:
  * strict separation between environment internals and agent observations;
  * no hidden rules / simulator state / evaluation labels inside policies;
  * bounded memory, deterministic seeds, crash-safe checkpoints;
  * CPU-only, Python standard library only (no GPU / cloud / paid APIs).
"""

__version__ = "0.1.0"
