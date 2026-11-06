#!/usr/bin/env python3
"""CORVID v0.6: paid *interventions* versus paid passive observations.

Synthetic diagnostic world: a latent binary actuator property determines whether
an intervention changes the actuator's state. Reading an unchanged sensor is a
passive observation. A pulse acts on the world, then a before/after contrast is
measured. Profiles and error rates are hidden from adaptive agents.

Evidence and feedback are budgeted; labels arrive after a fixed delay. The
oracle control has privileged simulator information and is NOT a fair peer.
No sensor efficacy or real energy claims are established by this experiment.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import random
import statistics

from adaptive_attention import BlockQuota, CONTEXTS, paired_ci

BLOCK=40
CONDITIONS=('stable','shift','iid','novel','sham')
POLICIES=('passive','probe','random','uncertainty','plug_in_error','bayes_voi','corvid','oracle')
WINDOW=64
ACTIONS=('passive','probe')


@dataclass(frozen=True)
class PublicView:
    context:int
    first:int


@dataclass(frozen=True)
class PrivateTrial:
    view:PublicView
    passive:int
    probe_before:int
    probe_after:int
    truth:int
    profile:int  # scorer and oracle only

    def reading(self, action:str)->int:
        if action=='passive': return self.passive
        if action=='probe': return self.probe_before ^ self.probe_after
        raise ValueError('unknown diagnostic')


# Error rates are generation-only and NOT visible to CORVID / other learned policies.
# Profiles: intervention helpful, passive helpful, passive helpful, intervention helpful.
PRIMARY=(.40,.08,.33,.24)
PASSIVE=(.38,.06,.08,.30)
PROBE=(.06,.48,.43,.10)


def world(condition:str,seed:int,trials:int)->list[PrivateTrial]:
    if condition not in CONDITIONS or trials < BLOCK or trials % BLOCK:
        raise ValueError('invalid environment or trial count')
    rng=random.Random(840173+seed*43319)
    mapping=list(range(CONTEXTS)); rng.shuffle(mapping)
    episodes=[]
    for i in range(trials):
        if i==trials//2 and condition=='shift':
            mapping=[mapping[(j+2)%CONTEXTS] for j in range(CONTEXTS)]
        ctx=rng.randrange(CONTEXTS)
        p=(rng.randrange(CONTEXTS) if condition=='iid' else mapping[ctx])
        y=rng.randrange(2)
        # Changing the generator in novel mode is a *synthetic* OOD analogue.
        if condition=='novel':
            p_error=(.35,.12,.45,.19)[p]
            passive_error=(.44,.11,.06,.37)[p]
            probe_error=(.12,.47,.45,.08)[p]
        else:
            p_error,passive_error,probe_error=PRIMARY[p],PASSIVE[p],PROBE[p]
        if condition=='sham':
            # The physical intervention has no usable causal signal here.
            # Diagnostic readout is independent of the hidden property.
            probe_error=.5
        # A correlated fault makes repeating the passive reading less helpful.
        shared=rng.random()<(.24 if p in (0,2) else .06)
        first=y ^ int(shared or rng.random()<p_error*(.70 if p in (0,2) else 1))
        passive=y ^ int(shared or rng.random()<passive_error*(.70 if p in (0,2) else 1))
        # Intervention: before/after difference is normally y (actuator
        # responsive), corrupted by regime-dependent measurement/effect noise.
        # The state change happens only because pulse=1. Without intervention
        # the counterfactual state stays at probe_before.
        before=rng.randrange(2)
        delta=y ^ int(rng.random()<probe_error)
        after=before ^ delta
        episodes.append(PrivateTrial(PublicView(ctx,first),passive,before,after,y,p))
    return episodes


class BoundedMemory:
    """Only *delivered verified labels* update bounded sensor statistics."""
    def __init__(self,window:int=WINDOW,decay:float=.96):
        if window < 1 or not (0<decay<=1):raise ValueError('bad memory configuration')
        self.window=window;self.decay=decay
        self.errors=[[[] for _ in range(3)] for _ in range(CONTEXTS)]
        self.cofailure=[[[] for _ in range(2)] for _ in range(CONTEXTS)]
        self.delivered=0

    def _push(self,history:list[int],value:int)->None:
        history.append(value)
        if len(history)>self.window:history.pop(0)

    def update_primary(self,context:int,primary:int,truth:int)->None:
        self._push(self.errors[context][0],int(primary!=truth))
        self.delivered+=1

    def update(self,context:int,primary:int,action:str,reading:int,truth:int)->None:
        idx=ACTIONS.index(action)
        self._push(self.errors[context][0],int(primary!=truth))
        self._push(self.errors[context][idx+1],int(reading!=truth))
        self._push(self.cofailure[context][idx],int(primary!=truth and reading!=truth))
        self.delivered+=1

    def _estimate(self,values:list[int],a:float,b:float)->float:
        n=len(values)
        w=[self.decay**(n-j-1) for j in range(n)]
        return (a+sum(x*y for x,y in zip(values,w)))/(a+b+sum(w))

    def error(self,context:int,sensor:int)->float:
        return self._estimate(self.errors[context][sensor],1,3)

    def count(self,context:int,action:str)->int:
        return len(self.errors[context][ACTIONS.index(action)+1])

    def predict(self,context:int,first:int,action:str|None=None,reading:int|None=None)->float:
        e0=max(.03,min(.97,self.error(context,0)))
        logit=(1 if first else -1)*math.log((1-e0)/e0)
        if action is not None:
            if reading is None:raise ValueError('reading is required')
            idx=ACTIONS.index(action)
            er=max(.03,min(.97,self.error(context,idx+1)))
            # Correlation term learned ONLY from labeled episodes, not oracle.
            joint=self._estimate(self.cofailure[context][idx],1,7)
            excess=max(0,joint-e0*er)
            scale=1-min(.8,2.5*excess)
            logit+=scale*(1 if reading else -1)*math.log((1-er)/er)
        return 1/(1+math.exp(-max(-40,min(40,logit))))

    def entries(self)->int:
        return sum(len(v) for row in self.errors for v in row)+sum(len(v) for row in self.cofailure for v in row)


@dataclass(frozen=True)
class Deferred:
    due:int
    context:int
    first:int
    action:str|None
    reading:int|None
    baseline:int
    final:int
    truth:int


class LearningAgent:
    def __init__(self,policy:str):
        self.policy=policy
        self.memory=BoundedMemory()
        self.credits=[[[0.0,0.0] for _ in ACTIONS] for _ in range(CONTEXTS)]
        self.counts=[[0 for _ in ACTIONS] for _ in range(CONTEXTS)]
        self.updates=0

    def choose(self,view:PublicView,rng:random.Random,profile:int|None=None)->str:
        ctx=view.context
        if self.policy=='passive':return 'passive'
        if self.policy=='probe':return 'probe'
        if self.policy=='random':return rng.choice(ACTIONS)
        if self.policy=='oracle':
            if profile is None:raise ValueError('privileged oracle profile required')
            return ('probe','passive','passive','probe')[profile]
        if self.policy=='uncertainty':
            return 'probe' if self.memory.error(ctx,0)>.27 else 'passive'
        if self.policy=='plug_in_error':
            if rng.random()<.12:return rng.choice(ACTIONS)
            return min(ACTIONS,key=lambda a:(self.memory.error(ctx,ACTIONS.index(a)+1),a))
        if self.policy=='bayes_voi':
            # Expected reduction in 0-1 Bayes risk under a simple conditional-
            # independence likelihood model. Its assumptions are wrong when
            # errors co-fail; that mismatch is visible in the benchmark.
            if rng.random()<.12:return rng.choice(ACTIONS)
            p=self.memory.predict(ctx,view.first)
            def expected_accuracy(action:str)->float:
                e=min(.97,max(.03,self.memory.error(ctx,ACTIONS.index(action)+1)))
                return (max(p*(1-e),(1-p)*e)
                        +max(p*e,(1-p)*(1-e)))
            estimates=[expected_accuracy(a) for a in ACTIONS]
            return ACTIONS[0] if estimates[0]>=estimates[1] else ACTIONS[1]
        if self.policy=='corvid':
            if rng.random()<.20:return rng.choice(ACTIONS)
            scores=[]
            for idx,_ in enumerate(ACTIONS):
                x,n=self.credits[ctx][idx]
                scores.append(x/(n+2) + .30/math.sqrt(1+n))
            return ACTIONS[0] if scores[0]>=scores[1] else ACTIONS[1]
        raise ValueError('unknown policy')

    def receive(self,d:Deferred)->None:
        if d.action is None:
            # No diagnostic occurred. The verified label informs only the
            # primary sensor, never a phantom passive measurement.
            self.memory.update_primary(d.context,d.first,d.truth)
        else:
            assert d.reading is not None
            self.memory.update(d.context,d.first,d.action,d.reading,d.truth)
            ix=ACTIONS.index(d.action)
            credit=int(d.final==d.truth)-int(d.baseline==d.truth)
            total,weight=self.credits[d.context][ix]
            self.credits[d.context][ix]=(.96*total+credit,.96*weight+1)
            self.counts[d.context][ix]+=1
        self.updates+=1


@dataclass(frozen=True)
class Metrics:
    condition:str
    policy:str
    seed:int
    trials:int
    accuracy:float
    post_shift_accuracy:float
    brier:float
    intervention_count:int
    passive_count:int
    paid_diagnostics:int
    requested_labels:int
    delivered_labels:int
    pending_labels:int
    memory_entries:int
    learning_updates:int
    intervention_helps:int  # scorer only
    intervention_harms:int  # scorer only


def simulate(condition:str,policy:str,seed:int,trials:int=800,
             diagnostic_fraction:float=.25,label_fraction:float=.10,
             delay:int=7)->Metrics:
    if policy not in POLICIES or delay<1:raise ValueError('invalid policy/delay')
    episodes=world(condition,seed,trials)
    agent=LearningAgent(policy)
    budget=BlockQuota(trials,diagnostic_fraction)
    labels=BlockQuota(trials,label_fraction)
    # Independent streams; same pre-generated worlds across all policies.
    slot_rng=random.Random(seed*7177+571)
    label_rng=random.Random(seed*2999+163)
    choice_rng=random.Random(seed*5897+233)
    pending:dict[int,list[Deferred]]=defaultdict(list)
    correct=[];brier=[];acted=0;passive=0;helps=0;harms=0;delivered=0
    for i,e in enumerate(episodes):
        for d in pending.pop(i,[]):
            agent.receive(d)
            delivered+=1
        view=e.view
        p0=agent.memory.predict(view.context,view.first)
        base=int(p0>.5) if p0!=.5 else view.first
        # No profile or truth passed into ordinary learned agents.
        choose_slot=budget.decide(i,1.0,1.0,slot_rng)
        action=None;reading=None
        if choose_slot:
            action=agent.choose(view,choice_rng,e.profile if policy=='oracle' else None)
            reading=e.reading(action)
            if action=='probe':acted+=1
            else:passive+=1
            p=agent.memory.predict(view.context,view.first,action,reading)
        else:p=p0
        yhat=int(p>.5) if p!=.5 else view.first
        # All methods request identical-label slots before truth is exposed.
        label=labels.decide(i,1.,1.,label_rng)
        if label:
            pending[i+delay].append(Deferred(i+delay,view.context,view.first,
                         action,reading,base,yhat,e.truth))
        if choose_slot:
            gain=int(yhat==e.truth)-int(base==e.truth)
            helps+=int(gain>0 and action=='probe')
            harms+=int(gain<0 and action=='probe')
        correct.append(int(yhat==e.truth));brier.append((p-e.truth)**2)
    outstanding=sum(len(p) for p in pending.values())
    assert delivered+outstanding==labels.used
    assert budget.used==trials*budget.per_block//BLOCK
    assert labels.used==trials*labels.per_block//BLOCK
    return Metrics(condition,policy,seed,trials,statistics.mean(correct),
            statistics.mean(correct[trials//2:trials//2+80]),statistics.mean(brier),
            acted,passive,budget.used,labels.used,delivered,outstanding,
            agent.memory.entries(),agent.updates,helps,harms)


def benchmark(seeds:int=30,trials:int=800,diagnostic_fraction:float=.25,
              label_fraction:float=.10,delay:int=7)->dict:
    if seeds<2:raise ValueError('at least two seeds required')
    data={}
    for condition in CONDITIONS:
        runs={p:[simulate(condition,p,s,trials,diagnostic_fraction,label_fraction,delay)
                 for s in range(seeds)] for p in POLICIES}
        data[condition]={
            'summary':{p:{key:statistics.mean(getattr(r,key) for r in rows)
                for key in ('accuracy','post_shift_accuracy','brier','intervention_count',
                            'passive_count','paid_diagnostics','requested_labels',
                            'delivered_labels','pending_labels','memory_entries',
                            'learning_updates')} for p,rows in runs.items()},
            'corvid_minus_bayes_voi':paired_ci([a.accuracy-b.accuracy for a,b in
                zip(runs['corvid'],runs['bayes_voi'])]),
            'corvid_minus_random':paired_ci([a.accuracy-b.accuracy for a,b in
                zip(runs['corvid'],runs['random'])]),
            'per_seed':{p:[asdict(r) for r in rows] for p,rows in runs.items()}
        }
    return {'protocol':{'version':'0.6','seeds':seeds,'trials':trials,
        'diagnostic_fraction':diagnostic_fraction,'label_fraction':label_fraction,
        'delay':delay,'diagnostic_action_cost':'one synthetic diagnostic slot; NOT equal physical joules',
        'probe_is_intervention':True,'labels_delayed':True,
        'oracle_uses_hidden_profile':True,'oracle_is_true_upper_bound':False,
        'bayesian_comparator_assumes_independent_errors':True,
        'learned_policies_access_truth':False,
        'paired_experiments':True,'real_energy_measured':False,
        'novel_architecture_proven':False,'safety_scope':'simulated bounded reversible actuator pulse'},
        'conditions':data}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds',type=int,default=30)
    ap.add_argument('--trials',type=int,default=800)
    ap.add_argument('--diagnostic-fraction',type=float,default=.25)
    ap.add_argument('--label-fraction',type=float,default=.10)
    ap.add_argument('--delay',type=int,default=7)
    ap.add_argument('--json',type=Path)
    a=ap.parse_args()
    result=benchmark(a.seeds,a.trials,a.diagnostic_fraction,a.label_fraction,a.delay)
    for c in CONDITIONS:
        print(f'\n{c:10s}  accuracy    diagnostics   pulses   labels')
        for p in POLICIES:
            row=result['conditions'][c]['summary'][p]
            print(f'{p:14s} {row["accuracy"]:7.2%} {row["paid_diagnostics"]:9.0f} '
                  f'{row["intervention_count"]:7.1f} {row["requested_labels"]:7.0f}')
        ci=result['conditions'][c]['corvid_minus_bayes_voi']
        print(f'CORVID - Bayes-VOI: {ci["mean"]:+.2%} '
              f'(95% bootstrap [{ci["ci95_low"]:+.2%}, {ci["ci95_high"]:+.2%}])')
    if a.json:
        a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Saved',a.json)
    print('\nSimulation only; physical cost equality and biological fidelity NOT established.')

if __name__=='__main__':main()
