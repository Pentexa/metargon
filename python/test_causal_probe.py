"""Reproducibility, hidden-state isolation, and cost/feedback invariants."""
import dataclasses
import math
import unittest
from causal_probe import (ACTIONS,CONDITIONS,POLICIES,BoundedMemory,Deferred,
                          LearningAgent,world,simulate,benchmark)


class CausalProbeTests(unittest.TestCase):
    def test_world_reproducible(self):
        a=world('stable',1,80)
        self.assertEqual(a,world('stable',1,80))
        self.assertNotEqual(a,world('stable',2,80))
        for trial in a:
            self.assertEqual(trial.reading('probe'),trial.probe_before ^ trial.probe_after)
            self.assertIn(trial.reading('passive'),(0,1))

    def test_reversible_perturbation_is_only_simulated(self):
        e=world('novel',2,80)[0]
        self.assertEqual(e.reading('probe'),e.probe_before ^ e.probe_after)
        self.assertEqual(e,world('novel',2,80)[0])
        with self.assertRaises(ValueError):e.reading('other')

    def test_bound_memory_after_many_deliveries(self):
        m=BoundedMemory(window=8)
        for i in range(200):
            m.update(0,1,ACTIONS[i%2],i%2,0)
        self.assertLessEqual(m.entries(),5*8)
        self.assertEqual(m.delivered,200)
        self.assertLessEqual(m.count(0,'passive'),8)
        self.assertLessEqual(m.count(0,'probe'),8)
        self.assertEqual(m.count(1,'passive'),0)

    def test_no_phantom_sensor_measurement(self):
        a=LearningAgent('corvid')
        a.receive(Deferred(7,0,1,None,None,1,1,0))
        self.assertEqual(a.memory.delivered,1)
        self.assertEqual(a.memory.count(0,'passive'),0)
        self.assertEqual(a.memory.count(0,'probe'),0)
        self.assertEqual(a.counts[0],[0,0])
        a.receive(Deferred(8,0,1,'probe',0,1,0,0))
        self.assertEqual(a.memory.count(0,'probe'),1)
        self.assertEqual(a.counts[0],[0,1])

    def test_same_world_and_exact_budgets(self):
        for condition in CONDITIONS:
            for policy in POLICIES:
                x=simulate(condition,policy,3,160)
                self.assertEqual(x.paid_diagnostics,40)
                self.assertEqual(x.intervention_count+x.passive_count,40)
                self.assertEqual(x.requested_labels,16)
                self.assertEqual(x.requested_labels,x.delivered_labels+x.pending_labels)
                self.assertEqual(x.learning_updates,x.delivered_labels)
                self.assertLessEqual(x.memory_entries,4*5*64)
                self.assertTrue(0<=x.accuracy<=1)
                self.assertTrue(0<=x.brier<=1)

    def test_no_feedback_no_learning(self):
        x=simulate('stable','corvid',0,160,label_fraction=0)
        self.assertEqual(x.delivered_labels,0)
        self.assertEqual(x.learning_updates,0)
        self.assertEqual(x.memory_entries,0)

    def test_no_lookahead_delays(self):
        # All requested labels remain pending if the delay exceeds the run.
        a=simulate('shift','corvid',1,160,delay=500)
        self.assertEqual(a.delivered_labels,0)
        self.assertEqual(a.pending_labels,16)
        self.assertEqual(a.memory_entries,0)

    def test_seed_reproducibility(self):
        a=simulate('shift','corvid',5,160)
        self.assertEqual(a,simulate('shift','corvid',5,160))
        self.assertNotEqual(a,simulate('shift','corvid',6,160))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):world('stable',0,41)
        with self.assertRaises(ValueError):world('bad',0,80)
        with self.assertRaises(ValueError):simulate('stable','bad',1,80)
        with self.assertRaises(ValueError):simulate('stable','random',1,80,delay=0)
        with self.assertRaises(ValueError):simulate('stable','random',1,80,label_fraction=.13)
        with self.assertRaises(ValueError):BoundedMemory(window=0)
        with self.assertRaises(ValueError):benchmark(seeds=1,trials=80)
        with self.assertRaises(ValueError):BoundedMemory().predict(0,1,'probe',None)

    def test_report_metadata_is_honest(self):
        report=benchmark(seeds=2,trials=80)
        p=report['protocol']
        self.assertTrue(p['oracle_uses_hidden_profile'])
        self.assertTrue(p['probe_is_intervention'])
        self.assertFalse(p['real_energy_measured'])
        self.assertFalse(p['novel_architecture_proven'])
        for condition in CONDITIONS:
            self.assertTrue(math.isfinite(report['conditions'][condition]['summary']['corvid']['accuracy']))

if __name__=='__main__':unittest.main()
