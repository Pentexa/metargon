#!/usr/bin/env python3
"""METARGON v0.8 illustrative visible control demo; no hidden truth sent to AI."""
from metargon_world import Episode, Policy, play_episode


def main():
    agent = Policy('metargon', 42)
    # These modes are evaluator-only annotations for the human demo; the
    # agent receives context ID + public action results only.
    modes = ['key', 'key', 'key', 'switch', 'switch', 'switch']
    print('METARGON v0.8 — rule change demonstration')
    print('The hidden mechanism below is displayed to the human, not the agent.\n')
    for idx, mode in enumerate(modes, 1):
        spec = Episode(context=0, mode=mode, sham=False,
                       uniforms={'key': (0.5,) * 4, 'switch': (0.5,) * 4})
        before = agent.posteriors[0]
        outcome = play_episode(spec, agent)
        first = outcome['attempts'][0]['recipe']
        print(f'Episode {idx}: evaluator rule={mode:>6}, belief_switch_before={before:.3f}, '
              f'first_attempt={first:>6}, reached_goal={outcome["success"]}, '
              f'steps={outcome["steps"]}, belief_switch_after={agent.posteriors[0]:.3f}')
        print('  observed actions:', ' '.join(outcome['trace']))
    print('\nCAUTION: controlled deterministic demo, not measured general intelligence.')


if __name__ == '__main__':
    main()
