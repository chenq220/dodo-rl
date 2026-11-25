import numpy as np
import pytest
import torch
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from algos.agents.input_shape import AgentInputShape
from algos.modules.agents.recurrent_controller import RecurrentController
from algos.modules.mixers.qmix import QMixer
from common.portal_entry import load_training_config
from config import EnvConfig, SpatialRepresentationConfig
from envs.sense.cooperative_search_env import CooperativeSearchEnv
from envs.state.representation import SpatialGridRepresentation
from sample.sample_transfer import EpisodeTransition

class OneStepBatch:
    def __init__(self, data):
        self.data = data
        self.batch_size = 1
        self.device = torch.device("cpu")

    def __getitem__(self, key):
        return self.data[key]



def test_spatial_representation_keeps_region_layout_and_batch_contract():
    encoder = SpatialGridRepresentation(
        SpatialRepresentationConfig(input_shape=(2, 4, 4), output_dim=7)
    )
    left = np.zeros((2, 4, 4), dtype=np.float32)
    right = left.copy()
    left[0, 0, 0] = 1
    right[0, 3, 3] = 1

    pair = encoder(np.stack((left, right)))
    assert pair.shape == (2, 7)
    assert not np.array_equal(pair[0], pair[1])
    np.testing.assert_array_equal(pair[0], encoder(left))
    assert np.isfinite(pair).all()


@pytest.mark.parametrize("output_dim", [0, -1, 1, 33, 1.5, True])
def test_spatial_representation_rejects_invalid_output_dim(output_dim):
    with pytest.raises(ValueError, match="output_dim"):
        SpatialGridRepresentation(
            SpatialRepresentationConfig(input_shape=(2, 4, 4), output_dim=output_dim)
        )


def test_spatial_representation_rejects_wrong_input_shape():
    encoder = SpatialGridRepresentation(
        SpatialRepresentationConfig(input_shape=(2, 4, 4), output_dim=7)
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        encoder(np.zeros((2, 5, 4), dtype=np.float32))


def test_full_length_spatial_representation_preserves_every_cell():
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    encoder = SpatialGridRepresentation(
        SpatialRepresentationConfig(input_shape=(2, 3, 4), output_dim=24)
    )
    np.testing.assert_array_equal(encoder(source), source.ravel())


def test_cooperative_search_representation_enters_obs_and_state():
    config = EnvConfig(
        sense_name="cooperative_search",
        use_sim=False,
        state_dim=0,
        agent_names=[f"drone_{i}" for i in range(4)],
        metadata={
            "game_config": {"num_dynamic_threats": 0},
            "dynamic_threats_enabled": False,
            "situation_representation_output_dim": 12,
        },
    )
    env = CooperativeSearchEnv(config)
    try:
        obs, _ = env.reset(seed=7)
        assert env.observation_dim == 487 + 12
        assert env.centralized_state_dim == 12 + 4 * 4
        original = obs["drone_0"].copy()
        state = env.state()
        np.testing.assert_array_equal(original[-12:], state[:12])

        # 相同覆盖格数、不同位置的图必须影响策略可见输入。
        env.game.positive_counts.fill(0)
        env.game.positive_counts[0, 0] = 1
        first = env.get_obs()["drone_0"][-12:]
        env.game.positive_counts.fill(0)
        env.game.positive_counts[-1, -1] = 1
        second = env.get_obs()["drone_0"][-12:]
        assert not np.array_equal(first, second)
        np.testing.assert_array_equal(second, env.state()[:12])

        with pytest.raises(Exception, match="cannot change"):
            env.apply_env_metadata({"situation_representation_output_dim": 16})
    finally:
        env.close()


def test_representation_qmix_subject_config_matches_environment():
    root = Path(__file__).resolve().parents[1]
    subject = root / "subjects/cooperative_search/cooperative_search_qmix_partitions.toml"
    conf = load_training_config(Namespace(config_file=str(subject)), root)
    agent = next(item for item in conf.step_agents_config if item.name == "uav_team")
    learner = next(item for item in conf.learners_config if item.name == "uav_team")
    from envs.sense.cooperative_search_partition_env import CooperativeSearchPartitionEnv
    env = CooperativeSearchPartitionEnv(conf.env_config)
    def group_state():
        return env.state_for_agent_ids(agent_name="uav_team",
            agent_ids=tuple(f"drone_{i}" for i in range(4)))
    try:
        assert conf.env_config.metadata["situation_representation_output_dim"] == 128
        assert agent.obs_dim == env.observation_dim == 615
        assert learner.state_shape == env.group_state_dim == 144
        assert env.centralized_state_dim == 608
        obs, _ = env.reset(seed=11)
        assert obs["drone_0"].shape == (615,)
        assert group_state().shape == (144,)

        agent.bind_runtime(n_agents=4, n_actions=12, state_shape=144, device="cpu")
        learner.bind_runtime(
            n_agents=4, n_actions=12, state_shape=144,
            obs_dim=615, hidden_dim=agent.hidden_dim, device="cpu",
        )
        input_shape = AgentInputShape(
            obs_dim=615, action_dim=12, n_agents=4,
            obs_last_action=agent.obs_last_action,
            obs_agent_id=agent.obs_agent_id,
        )
        mac = RecurrentController(
            scheme=input_shape.to_scheme(), groups={"agents": 4}, args=agent,
        )
        mixer = QMixer(4, 144, learner.algo_config.mixing_embed_dim,
                       learner.algo_config.hypernet_layers, learner.algo_config.hypernet_embed)
        obs_tensor = torch.from_numpy(np.stack([obs[f"drone_{i}"] for i in range(4)]))
        batch = OneStepBatch(
            data={
                "obs": obs_tensor[None, None],
                "actions_onehot": torch.zeros((1, 1, 4, 12)),
                "avail_actions": torch.ones((1, 1, 4, 12)),
            },
        )
        mac.init_hidden(batch_size=1)
        q = mac.forward(batch, t=0)
        mixed = mixer(q.max(dim=-1).values[:, None], torch.from_numpy(group_state())[None, None])
        mixed.mean().backward()
        assert q.shape == (1, 4, 12)
        assert mixed.shape == (1, 1, 1)
        assert torch.isfinite(mac.agent.fc1.weight.grad).all()

        initial_state = group_state()
        env.game.positive_counts[0, 0] += 1
        next_obs = env.get_obs()
        next_state = group_state()
        drones = [f"drone_{i}" for i in range(4)]
        transition = EpisodeTransition()
        transition.add_step(
            obs={name: obs[name] for name in drones},
            action={name: SimpleNamespace(action_idx=0) for name in drones},
            last_action={name: None for name in drones},
            reward={name: 0.0 for name in drones},
            next_obs={name: next_obs[name] for name in drones},
            termination={name: False for name in drones},
            truncation={name: False for name in drones},
            state=initial_state,
            next_state=next_state,
            avail_actions_next={name: [1] * 12 for name in drones},
        )
        replay_batch = EpisodeTransition.to_batch(drones, [transition], action_dim=12)
        assert replay_batch["obs"].shape == (1, 2, 4, 615)
        assert replay_batch["state"].shape == (1, 2, 144)
        np.testing.assert_array_equal(
            replay_batch["obs"][0, 1, 0, -128:].numpy(), next_obs["drone_0"][-128:]
        )
    finally:
        env.close()


def test_original_qmix_subject_retains_unencoded_observation_and_state():
    root = Path(__file__).resolve().parents[1]
    subject = root / "subjects/cooperative_search/cooperative_search_qmix.toml"
    conf = load_training_config(Namespace(config_file=str(subject)), root)
    env = CooperativeSearchEnv(conf.env_config)
    try:
        assert env.situation_representation is None
        assert env.observation_dim == 487
        assert env.centralized_state_dim == 5028
        obs, _ = env.reset(seed=11)
        assert obs["drone_0"].shape == (487,)
        assert env.state().shape == (5028,)
    finally:
        env.close()
