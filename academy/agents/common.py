"""METARGON Academy — shared agent helpers.

Everything here consumes only public observations and public geometry.
No hidden simulator state is ever referenced.
"""

from __future__ import annotations

from ..envs.gridworld import (
    ACTIONS, DOORS, PLATES, SWITCHES, WS, public_next,
    atoms_from_state, make_ws,
)


def atoms_from_observation(obs) -> frozenset:
    """Extract true grounded atoms from a public observation."""
    return atoms_from_state(obs.plate_load, obs.switch_state)


def belief_from_obs(obs, geom) -> WS:
    """Rebuild the belief state from a public observation (perception)."""
    b = make_ws(geom)
    b = b._replace(pos=(obs.x, obs.y))
    objs = dict(b.objs)
    objs["A"] = (obs.ax, obs.ay)
    objs["B"] = (obs.bx, obs.by)
    b = b._replace(objs=objs)
    sw = dict(b.switch_state)
    sw["S1"] = obs.s1
    sw["S2"] = obs.s2
    b = b._replace(switch_state=sw)
    pl = dict(b.plate_load)
    pl["P1"] = obs.p1
    pl["P2"] = obs.p2
    b = b._replace(plate_load=pl)
    doors = dict(b.door_open)
    doors["D1"] = obs.d1
    doors["D2"] = obs.d2
    b = b._replace(door_open=doors)
    return b


def valid_actions(state: WS, geom) -> list:
    return [a for a in ACTIONS if public_next(state, a, geom) is not None]


def sign(obs):
    """Compact state signature for episodic memory."""
    return (obs.x, obs.y, obs.ax, obs.ay, obs.bx, obs.by,
            obs.s1, obs.s2, obs.p1, obs.p2, obs.d1, obs.d2)


def state_key(st: WS):
    return (st.pos, st.objs["A"], st.objs["B"],
            tuple(st.switch_state[s] for s in SWITCHES),
            tuple(st.plate_load[p] for p in PLATES),
            tuple(st.door_open[d] for d in DOORS))
