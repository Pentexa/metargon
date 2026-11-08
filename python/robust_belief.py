"""METARGON v0.10 — robust reconciled belief from noisy public observations.

The public observation channel can corrupt switch / plate / door readings
each step independently (probability q). A v0.9-era agent copies readings
straight into its belief, so a single corrupted bit can (a) fabricate a
door event, (b) corrupt event contexts used for causal learning, and
(c) corrupt planning (invalid actions into really-open/closed doors).

This module rebuilds a *physical* belief using only family-agnostic,
public knowledge:

  1. positions and object locations are never corrupted -> copied directly.
  2. the agent's own deterministic action explains some sensor changes
     exactly (Toggle flips a switch we stand on; Push moves an object
     onto/off a plate). Such readings are adopted immediately.
  3. a move that fails (invalid action) proves the target cell is blocked:
     a wall (fixed and known) or a closed door -> adopt `closed` for that
     door.
  4. a move that succeeds into a door cell proves that door is open.
  5. every other reading change must persist for DEBOUNCE consecutive
     observations before it is believed (independent per-step noise is
     extremely unlikely to persist; real mechanisms persist until changed).

No hidden state, no family knowledge.
"""
from __future__ import annotations

from lab_env import DOORS, DOOR_CELL, MOVE_ACTIONS, PLATES, SWITCHES, WS, make_state

# pre-registered: a correction becomes believed after this many consecutive
# agreeing corrupted readings (holds q up to ~0.2 without meaningful cost).
DEBOUNCE = 2
# extra agreeing raw reading (beyond the first flip) needed to *confirm* a
# door event for the learning stream (kept separate from the belief, which
# still adopts immediately so planning never lags).
CONFIRM_PERSIST = 1

_DIR = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


class RobustBelief:
    """Maintain a reconciled WS belief from the public observation stream."""

    def __init__(self, seed: int = 0, debounce: int = DEBOUNCE):
        self.debounce = debounce
        self.belief = make_state()
        self.pending: dict[str, tuple] = {}   # sensor -> (value, count)
        self.noise_anomalies = 0              # diagnostics
        self.action_corrections = 0           # diagnostics (physics-adopted)
        self.sensor_changes = 0               # debounce-adopted changes
        # -- noise-aware confirmed-event stream --------------------------
        # raw readings are tracked separately from the belief; a door event
        # is *confirmed* for learning when either
        #   (a) the flipped raw reading persists on the next observation, or
        #   (b) movement proof confirms it (apply_transition), or
        #   (c) it reverts immediately -> candidate is dropped silently.
        self.raw_prev: dict[str, bool] = {d: False for d in DOORS}
        self.candidates: dict[str, tuple] = {}   # door -> (value, persist_count)
        self.confirmed_events: list[tuple] = []
        self.confirmed_count = 0
        self.dropped_candidates = 0
        self.confirmed_by_movement = 0
        self._last_confirmed: dict[str, tuple] = {}  # door -> (value, clock)
        self._clock = 0

    # -- diagnostics -----------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "noise_anomalies": self.noise_anomalies,
            "action_corrections": self.action_corrections,
            "sensor_changes": self.sensor_changes,
            "pending": len(self.pending),
            "confirmed_events": self.confirmed_count,
            "confirmed_by_movement": self.confirmed_by_movement,
            "dropped_candidates": self.dropped_candidates,
            "open_candidates": len(self.candidates),
        }

    # -- core ------------------------------------------------------------
    def observe(self, obs) -> WS:
        """Fold one observation into the belief (readings + positions)."""
        self._clock += 1
        b = self.belief
        b = b._replace(pos=(obs.x, obs.y))
        objs = dict(b.objs)
        objs["A"] = (obs.ax, obs.ay)
        objs["B"] = (obs.bx, obs.by)
        b = b._replace(objs=objs)

        sw = dict(b.switch_state)
        for name in SWITCHES:
            raw = getattr(obs, name.lower())
            self._debounce(name, raw, sw, (name in ("S1", "S2")))
        b = b._replace(switch_state=sw)

        pl = dict(b.plate_load)
        for name in PLATES:
            raw = getattr(obs, name.lower())
            self._debounce(name, raw, pl, False)
        b = b._replace(plate_load=pl)

        doors = dict(b.door_open)
        for name in DOORS:
            raw = getattr(obs, name.lower())
            if raw != doors[name]:
                # Doors are adopted immediately: a wrong belief self-corrects
                # cheaply via movement proof (invalid moves into closed doors,
                # successful moves through open doors) and probe outcomes.
                # Debouncing here adds lag that loses genuinely rapid
                # open/close transitions.
                doors[name] = raw
                self.sensor_changes += 1
                self.pending.pop(name, None)
            else:
                self.pending.pop(name, None)
            # confirmed-event bookkeeping (learning stream)
            if name in self.candidates:
                cand_value, persist = self.candidates[name]
                if raw == cand_value:
                    if persist >= CONFIRM_PERSIST:
                        self._emit_confirmed(name, bool(cand_value))
                        del self.candidates[name]
                    else:
                        self.candidates[name] = (cand_value, persist + 1)
                else:
                    # revert/inconsistent before confirmation: drop
                    del self.candidates[name]
                    self.dropped_candidates += 1
            elif raw != self.raw_prev[name]:
                self.candidates[name] = (raw, 0)
            self.raw_prev[name] = raw
        b = b._replace(door_open=doors)

        self.belief = b
        return b

    def apply_transition(self, action, valid, obs_after) -> WS:
        """Adopt states proven by the agent's own action (public physics).
        Must be called after observe(obs_after) for this step."""
        b = self.belief
        if action is None:
            return b

        if action == "Toggle":
            # standing on a switch: toggling flips it deterministically
            for name in SWITCHES:
                if (obs_after.x, obs_after.y) == _switch_cell(name):
                    want = not self._pre_action_switch(name, obs_after)
                    b.switch_state[name] = want
                    self.pending.pop(name, None)
                    self.action_corrections += 1
            return b

        if action in MOVE_ACTIONS:
            dx, dy = _DIR[action]
            pre_pos = self._pre_action_pos(action, obs_after)
            target = (pre_pos[0] + dx, pre_pos[1] + dy)
            if target in DOOR_CELL.values():
                door = next(d for d, c in DOOR_CELL.items() if c == target)
                old = b.door_open[door]
                if not valid:
                    if old:
                        self.action_corrections += 1
                    b.door_open[door] = False
                    self._confirm_by_movement(door, False, changed=(old is not False))
                else:
                    if not old:
                        self.action_corrections += 1
                    b.door_open[door] = True
                    self._confirm_by_movement(door, True, changed=(old is not True))
                self.pending.pop(door, None)
                self.pending.pop(door.lower(), None)
            return b

        if action.startswith("Push"):
            dx, dy = _DIR[action[4:]]
            pre_pos = self._pre_action_pos(action, obs_after)
            c1 = (pre_pos[0] + dx, pre_pos[1] + dy)
            c2 = (c1[0] + dx, c1[1] + dy)
            for pname, pc in _plate_cells().items():
                if pc == c1:
                    b.plate_load[pname] = "none"
                    self.pending.pop(pname, None)
                if pc == c2:
                    if (obs_after.ax, obs_after.ay) == c2:
                        b.plate_load[pname] = "A"
                    elif (obs_after.bx, obs_after.by) == c2:
                        b.plate_load[pname] = "B"
                    self.pending.pop(pname, None)
            return b
        return b

    # -- helpers ---------------------------------------------------------
    def _emit_confirmed(self, door, value):
        """Add a confirmed event unless the same direction was just confirmed
        (dedupe window) — a door transition is a single causal event."""
        last = self._last_confirmed.get(door)
        if last is not None and last[0] == value and \
                self._clock - last[1] < 3:
            return
        self.confirmed_events.append((door, bool(value)))
        self.confirmed_count += 1
        self._last_confirmed[door] = (bool(value), self._clock)

    def _confirm_by_movement(self, door, value, changed: bool = True):
        """Movement proof confirms a door state. An event is emitted only
        when the belief actually changed (`changed`) or a raw candidate was
        pending — repeated transits through an already-known state must not
        spam the learning stream with duplicate causal events."""
        had_candidate = door in self.candidates
        if not changed and not had_candidate:
            return
        self._emit_confirmed(door, value)
        self.confirmed_by_movement += 1
        self.candidates.pop(door, None)

    def poll_confirmed_events(self):
        """Drain confirmed door events since the last poll (learning stream)."""
        ev = list(self.confirmed_events)
        self.confirmed_events.clear()
        return ev

    def has_door_candidate(self, door) -> bool:
        return door in self.candidates

    def door_is_stable(self, door) -> bool:
        """A door is stable when there is no pending candidate and the raw
        reading agrees with the belief."""
        return door not in self.candidates and \
            self.raw_prev.get(door) == self.belief.door_open.get(door)

    def sensor_is_stable(self, sensor) -> bool:
        """Switch/plate sensor is stable when it has no pending (debounce)
        entry."""
        return sensor not in self.pending

    def stable_atoms(self) -> frozenset:
        """Atoms from sensors that are currently stable (not pending, not
        mid-event). Used for certainty-aware learning contexts."""
        out = set()
        for plate in PLATES:
            load = self.belief.plate_load[plate]
            if load in ("A", "B") and self.sensor_is_stable(plate):
                out.add(("load", plate, load))
                out.add(("loadany", plate))
        for sw in SWITCHES:
            if self.sensor_is_stable(sw):
                out.add(("sw", sw, bool(self.belief.switch_state[sw])))
        for door in DOORS:
            if self.door_is_stable(door):
                out.add(("door_open", door, bool(self.belief.door_open[door])))
        return frozenset(out)

    def context_certainty(self, atoms) -> float:
        """Fraction of the context atoms whose sensors were stable (floor
        0.25 so a partially noisy context never fully loses its learning
        signal)."""
        if not atoms:
            return 1.0
        stable = 0
        for a in atoms:
            k = a[0]
            if k in ("load", "loadany"):
                stable += int(self.sensor_is_stable(a[1]))
            elif k == "sw":
                stable += int(self.sensor_is_stable(a[1]))
            elif k == "door_open":
                stable += int(self.door_is_stable(a[1]))
            else:
                stable += 1
        return max(0.25, stable / len(atoms))

    def _debounce(self, key, raw, container, is_switch):
        cur = container[key]
        if raw == cur:
            self.pending.pop(key, None)
            return
        cand, cnt = self.pending.get(key, (raw, 0))
        if cand != raw:
            cnt = 0
        cnt += 1
        if cnt >= self.debounce:
            container[key] = raw
            self.sensor_changes += 1
            self.pending.pop(key, None)
        else:
            self.pending[key] = (raw, cnt)

    def _pre_action_pos(self, action, obs_after):
        """Position before the action, recovered from the new observation."""
        dx, dy = _DIR[action] if action in MOVE_ACTIONS else _DIR[action[4:]]
        return (obs_after.x - dx, obs_after.y - dy)

    def _pre_action_switch(self, name, obs_after):
        """The switch reading as it was *before* the Toggle. The raw
        post-toggle reading can be corrupted, so we rely on the toggle
        being deterministic: pre = toggled(post-belief). We cannot know
        pre from the reading alone; we trust our prior belief.
        """
        return self.belief.switch_state[name]


def _switch_cell(name):
    from lab_env import SWITCH_CELL
    return SWITCH_CELL[name]


def _plate_cells():
    from lab_env import PLATE_CELL
    return PLATE_CELL
