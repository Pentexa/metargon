"""Tests for the adaptive task selector (learning progress + weakness)."""

from __future__ import annotations

import random
import unittest

from academy.envs.gridworld import EnvConfig
from academy.envs.tasks import EpisodeSpec
from academy.evren.selection import (
    DEFAULTS, FamilyStats, LearningProgressSelector, WeaknessTracker,
    atom_class, atom_classes_of, family_key, rule_structure,
)
from academy.envs.tasks import difficulty_of


def _spec(cfg, seed=0):
    return EpisodeSpec(config=cfg, max_steps=8, noise_q=0.0,
                       spec_seed=seed, pool_tag="train",
                       difficulty=difficulty_of(cfg))


C_SW_TRUE = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
                      frozenset({("load", "P2", "B")}), False, False)
C_SW_FALSE = EnvConfig("TP1", frozenset({("sw", "S2", False)}),
                       frozenset({("load", "P2", "B")}), False, False)
C_EMPTY = EnvConfig("TP1", frozenset({("empty", "P1")}),
                    frozenset({("load", "P2", "B")}), False, False)


class TestStructure(unittest.TestCase):
    def test_atom_class_polarity_and_kind(self):
        self.assertEqual(atom_class(("sw", "S1", True)), ("sw", True))
        self.assertEqual(atom_class(("sw", "S2", False)), ("sw", False))
        self.assertEqual(atom_class(("load", "P1", "A")), ("load", "A"))
        self.assertEqual(atom_class(("loadany", "P2")), ("loadany",))
        self.assertEqual(atom_class(("empty", "P1")), ("empty",))

    def test_family_key_clause_order_invariant(self):
        a = EnvConfig("TP2", frozenset({("sw", "S2", True)}),
                      (frozenset({("sw", "S1", False)}),
                       frozenset({("empty", "P2")})), True, False)
        b = EnvConfig("TP2", frozenset({("sw", "S2", True)}),
                      (frozenset({("empty", "P2")}),
                       frozenset({("sw", "S1", False)})), True, False)
        self.assertEqual(family_key(a), family_key(b))

    def test_family_key_polarity_sensitive(self):
        self.assertNotEqual(family_key(C_SW_TRUE), family_key(C_SW_FALSE))

    def test_family_key_door_swap_collapses(self):
        a = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
                      frozenset({("load", "P2", "B")}), False, False)
        b = EnvConfig("TP1", frozenset({("load", "P2", "B")}),
                      frozenset({("sw", "S2", True)}), False, False)
        self.assertEqual(family_key(a), family_key(b))

    def test_family_key_delay_insensitive(self):
        a = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
                      frozenset({("load", "P2", "B")}), False, True)
        b = EnvConfig("TP1", frozenset({("sw", "S2", True)}),
                      frozenset({("load", "P2", "B")}), True, False)
        self.assertEqual(family_key(a), family_key(b))

    def test_family_key_is_json_roundtrippable(self):
        from academy.evren.selection import family_key_json, _json_key_to_family
        fk = family_key(C_SW_TRUE)
        self.assertEqual(_json_key_to_family(family_key_json(fk)), fk)

    def test_atom_classes_of_union(self):
        cls = atom_classes_of(C_EMPTY)
        self.assertIn(("empty",), cls)
        self.assertIn(("load", "B"), cls)


class TestFamilyStats(unittest.TestCase):
    def test_min_visits_gate(self):
        st = FamilyStats(window=10, min_visits=3)
        for x in (True, False, True):
            st.record(x, 0)
        self.assertIsNone(FamilyStats(window=10, min_visits=3).success_rate())

    def test_learning_progress_tracks_change(self):
        st = FamilyStats(window=6, min_visits=2)
        seq = [False, False, False, True, True, True]
        for i, x in enumerate(seq):
            st.record(x, i)
        self.assertAlmostEqual(st.success_rate(), 0.5)
        lp = st.learning_progress()
        self.assertAlmostEqual(lp, 1.0)   # 0% -> 100% between window halves

    def test_weakness_nonnegative(self):
        st = FamilyStats(window=4, min_visits=2)
        st.record(False, 0)
        st.record(False, 1)
        self.assertEqual(st.weakness(0.75), 0.75)
        self.assertEqual(st.weakness(0.0), 0.0)


class TestWeaknessTracker(unittest.TestCase):
    def test_records_and_summary(self):
        tr = WeaknessTracker(window=6, min_visits=2, target=0.75)
        tr.record(C_EMPTY, False, 1)
        tr.record(C_EMPTY, False, 2)
        tr.record(C_SW_TRUE, True, 3)
        tr.record(C_SW_TRUE, True, 4)
        rows = tr.summary()
        by = {tuple(r["atom_class"]): r for r in rows}
        self.assertGreater(by[("empty",)]["weakness"],
                           by[("sw", True)]["weakness"])

    def test_unknown_class_no_boost(self):
        tr = WeaknessTracker(window=6, min_visits=2, target=0.75)
        self.assertEqual(tr.weakness_of(("sw", True)), 0.0)


class TestSelector(unittest.TestCase):
    def test_rejects_unknown_params(self):
        with self.assertRaises(ValueError):
            LearningProgressSelector([], nonsense=1)

    def test_deterministic_selection_sequence(self):
        cands = [C_SW_TRUE, C_SW_FALSE, C_EMPTY]
        s1 = LearningProgressSelector(cands, seed=5)
        s2 = LearningProgressSelector(cands, seed=5)
        r1 = [s1.select(random.Random(9))[0] for _ in range(12)]
        r2 = [s2.select(random.Random(9))[0] for _ in range(12)]
        self.assertEqual(r1, r2)

    def test_lp_prefers_improving_family(self):
        cands = [C_SW_TRUE, C_SW_FALSE]
        sel = LearningProgressSelector(cands, seed=0, min_visits=2,
                                       noise_weight=0.0, window=8)
        # both families sit at the same measured success (~50% => equal
        # weakness); family(sw=True) is *improving* 0% -> 100%, while
        # family(sw=False) stalls at 50% (zero learning progress).
        seq_a = [False, False, False, False, True, True, True, True]
        seq_b = [True, False, True, False, True, False, True, False]
        for i in range(8):
            sel.record(_spec(C_SW_TRUE, i), seq_a[i])
            sel.record(_spec(C_SW_FALSE, i), seq_b[i])
        self.assertAlmostEqual(sel.family_score(family_key(C_SW_TRUE))
                               - sel.family_score(family_key(C_SW_FALSE)),
                               1.0, places=6)   # +lp_weight * |delta|=1.0

    def test_weakness_boost_for_failing_family(self):
        cands = [C_SW_TRUE, C_SW_FALSE]
        sel = LearningProgressSelector(cands, seed=0, min_visits=2,
                                       noise_weight=0.0)
        for i in range(4):
            sel.record(_spec(C_SW_TRUE, i), True)     # mastered (weakness 0)
            sel.record(_spec(C_SW_FALSE, i), False)   # failing
        self.assertGreater(
            sel.family_score(family_key(C_SW_FALSE)),
            sel.family_score(family_key(C_SW_TRUE)))

    def test_novelty_bonus_before_evidence(self):
        cands = [C_SW_TRUE, C_SW_FALSE]
        sel = LearningProgressSelector(cands, seed=0, min_visits=3,
                                       novelty_weight=0.8)
        sel.record(_spec(C_SW_TRUE, 0), False)
        self.assertEqual(sel.family_score(family_key(C_SW_FALSE)),
                         DEFAULTS["novelty_weight"])

    def test_state_roundtrip_keeps_scores(self):
        cands = [C_SW_TRUE, C_SW_FALSE, C_EMPTY]
        sel = LearningProgressSelector(cands, seed=4)
        rng = random.Random(1)
        for _ in range(10):
            idx, _ = sel.select(rng)
            sel.record(_spec(cands[idx], 0), bool(rng.getrandbits(1)))
        saved = sel.save_state()
        sel2 = LearningProgressSelector(cands, seed=999)
        sel2.load_state(saved)
        self.assertAlmostEqual(
            sel.family_score(family_key(C_SW_TRUE)),
            sel2.family_score(family_key(C_SW_TRUE)))
        self.assertEqual(sel._episodes, sel2._episodes)
        self.assertEqual(len(sel.family_stats), len(sel2.family_stats))

    def test_record_outside_pool_is_kept(self):
        cands = [C_SW_TRUE]
        sel = LearningProgressSelector(cands, seed=0)
        sel.record(_spec(C_EMPTY, 0), False)          # foreign family
        self.assertIn(family_key(C_EMPTY), sel.family_stats)
        self.assertIn(("empty",), {tuple(r["atom_class"])
                                   for r in sel.weakness.summary()})

    def test_progress_report_shape(self):
        sel = LearningProgressSelector([C_SW_TRUE], seed=0)
        sel.record(_spec(C_SW_TRUE, 0), True)
        rep = sel.progress_report()
        self.assertIn("families", rep)
        self.assertIn("weakness", rep)
        self.assertEqual(rep["episodes"], 1)


if __name__ == "__main__":
    unittest.main()
