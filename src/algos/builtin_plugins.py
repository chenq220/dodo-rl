"""Built-in algorithm entry points shared by Portal and the SDK."""
from types import MappingProxyType


BUILTIN_AGENT_PLUGINS = MappingProxyType({
    "rnn": "algos.modules.agents.rnn_agent:RNNAgent",
    "policy_rnn": "algos.modules.agents.policy_agent:PolicyRNNAgent",
    "q_rnn": "algos.modules.agents.q_agent:QAgent",
    "mappo_rnn": "algos.modules.agents.policy_agent:PolicyRNNAgent",
    "rnn_ns": "algos.modules.agents.rnn_ns_agent:RNNNSAgent",
    "monte_carlo": "algos.modules.agents.random_agent:RandomAgent",
    "sb3": "algos.modules.agents.sb3_agent:SB3Agent",
    "maddpg_mlp": "algos.modules.agents.maddpg_agent:MaddpgAgent",
})

BUILTIN_LEARNER_PLUGINS = MappingProxyType({
    "qmix": "algos.training.qmix_learner:QMixerLearner",
    "mappo": "algos.training.mappo_learner:MAPPOActorCriticLearner",
    "ppo": "algos.training.sb3_ppo_learner:SB3PPOLearner",
    "a2c": "algos.training.sb3_a2c_learner:SB3A2CLearner",
    "dqn": "algos.training.sb3_dqn_learner:SB3DQNLearner",
    "coma": "algos.training.coma_learner:COMALearner",
    "maddpg": "algos.training.maddpg_learner:MADDPGLearner",
})
