# MAPPO implementation provenance

Source: https://github.com/marlbenchmark/on-policy
Commit: de66d7a4b23fac2513f56f96f73b3f5cb96695ac
License: MIT (see the upstream repository).

rnn.py is the upstream RNNLayer. The surrounding actor, centralized critic,
GAE and clipped PPO update follow the upstream recurrent MAPPO design, adapted
to DodoRL EpisodeTransition, plugins, model publication and snapshots.
This is an adaptation, not a drop-in copy of the upstream trainer.

Changes: Gymnasium layouts; independent categorical heads with joint log probability;
tanh-transformed Gaussian with Jacobian correction for normalized bounded actions;
behavior log probabilities recorded by rollout; padded episode batches with true
termination versus time-limit bootstrap; entire episodes as recurrent minibatches.
QMIX and COMA are provided by native DodoRL learners as of 0.17.0.
