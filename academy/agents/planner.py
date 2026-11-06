"""METARGON Academy — bounded planner over public physics.

The planner never sees hidden rules. It plans over the public mechanics
(movement, pushing, toggling) toward either a target cell or a target set
of ground atoms, using only the agent's *belief* about doors. A connected-
component feasibility filter prunes candidate antecedents cheaply.
"""

from __future__ import annotations

from collections import deque

from .common import state_key
from ..envs.gridworld import (
    ACTIONS, DOORS, PLATES, SWITCHES, WS, atoms_from_state, public_next,
)


def atoms_of_state(st: WS) -> frozenset:
    return atoms_from_state(st.plate_load, st.switch_state)


def plan_to_pos(state: WS, target_pos, geom, max_expand: int = 1500,
                on_expand=None):
    """Shortest plan (public physics only) to occupy target_pos."""
    if state.pos == target_pos:
        return []
    start_key = state_key(state)
    queue = deque([(state, [])])
    seen = {start_key}
    while queue and len(seen) < max_expand:
        st, path = queue.popleft()
        for a in ACTIONS:
            nxt = public_next(st, a, geom)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if on_expand is not None:
                on_expand()
            np = path + [a]
            if nxt.pos == target_pos:
                return np
            queue.append((nxt, np))
    return None


def plan_to_atoms(state: WS, target: frozenset, geom,
                  max_expand: int = 1500, on_expand=None):
    """Shortest plan to a state whose atoms are a superset of `target`."""
    if target <= atoms_of_state(state):
        return []
    start_key = state_key(state)
    queue = deque([(state, [])])
    seen = {start_key}
    while queue and len(seen) < max_expand:
        st, path = queue.popleft()
        for a in ACTIONS:
            nxt = public_next(st, a, geom)
            if nxt is None:
                continue
            k = state_key(nxt)
            if k in seen:
                continue
            seen.add(k)
            if on_expand is not None:
                on_expand()
            np = path + [a]
            if target <= atoms_of_state(nxt):
                return np
            queue.append((nxt, np))
    return None


def components(state: WS, geom):
    """Connected components of walkable cells under the door belief
    (objects ignored). Returns a dict cell -> component id."""
    comp = {}
    cid = 0
    walkable = [(x, y)
                for x in range(7) for y in range(5)
                if not geom.cell_blocked((x, y), state.door_open)]
    for cell in walkable:
        if cell in comp:
            continue
        stack = [cell]
        comp[cell] = cid
        while stack:
            sx, sy = stack.pop()
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                n = (sx + dx, sy + dy)
                if n in walkable and n not in comp:
                    comp[n] = cid
                    stack.append(n)
        cid += 1
    return comp


def target_feasible(state: WS, target: frozenset, geom) -> bool:
    """Cheap component-based feasibility: reject antecedents whose atoms
    are mutually unreachable under the current door belief."""
    comp = components(state, geom)
    agent = state.pos
    if agent not in comp:
        return False
    c_agent = comp[agent]
    for atom in target:
        k = atom[0]
        if k in ("load", "loadany"):
            plate = atom[1]
            pc = geom.plate_cell[plate]
            if pc not in comp:
                return False
            if k == "load":
                obj = atom[2]
                oc = state.objs[obj]
                if oc not in comp or comp[oc] != comp[pc] or comp[oc] != c_agent:
                    return False
            elif comp[pc] != c_agent:
                return False
        elif k == "empty":
            plate = atom[1]
            pc = geom.plate_cell[plate]
            if pc not in comp or comp[pc] != c_agent:
                return False
        elif k == "sw":
            sc = geom.switch_cell[atom[1]]
            if sc not in comp or comp[sc] != c_agent:
                return False
    return True
