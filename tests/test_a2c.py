from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from algos.map import AlogsMap
from common.creator import create_agent, create_learner
from common.plugins import PluginsConfig, activate_plugins
from config import A2CConfig, AgentConfig, LearnerConfig
from portal.plugins.registry import project_plugin_registry
from envs.action.model import ActionParameterDefined, StepActionDefined
from roles.common.config_alignment import align_agent_config_with_learner_config
from sample.sample_transfer import SAEpisodeTransition
from utils.model_weights import snapshot_model_weights


def _build_a2c_pair():
    activate_plugins(PluginsConfig(), default_plugins=project_plugin_registry())
    learner_config = LearnerConfig(name="solo", algo_type=AlogsMap.a2c.learner)
    agent_config = AgentConfig(name="solo", agent_type=AlogsMap.a2c.agent.sb3)
    assert isinstance(learner_config.algo_config, A2CConfig)

    learner_config.device = "cpu"
    learner_config.algo_config.batch_size = 2
    agent_config.device = "cpu"
    agent_config.agent_names = ["solo-0"]
    agent_config.obs_dim = 3
    agent_config.define_action = [
        StepActionDefined(
            action_key="choose",
            parameters=[ActionParameterDefined(key="mode", values=[0, 1])],
            add_target=False,
        )
    ]
    align_agent_config_with_learner_config(agent_config, learner_config)

    agent = create_agent(
        agent_type=agent_config.agent_type,
        agent_config=agent_config,
        agent_net_source="blanco",
        agent_list=agent_config.agent_names,
        device=agent_config.device,
    )
    learner = create_learner(
        learner_config=learner_config,
        agent=agent,
        env=None,
        device=learner_config.device,
        net_source="blanco",
    )
    return agent, learner


def _build_episode(episode_num: int, *, terminated_at_end: bool) -> SAEpisodeTransition:
    episode = SAEpisodeTransition()
    episode.episode_num = episode_num
    for step in range(4):
        obs = np.asarray([episode_num, step, step % 2], dtype=np.float32)
        next_obs = np.asarray([episode_num, step + 1, (step + 1) % 2], dtype=np.float32)
        episode.add_step(
            obs=obs,
            action=step % 2,
            reward=1.0 if step % 2 == 0 else -0.25,
            next_obs=next_obs,
            terminated=terminated_at_end and step == 3,
            truncated=not terminated_at_end and step == 3,
            last_action=(step - 1) % 2 if step > 0 else None,
            avail_actions=[1, 1],
        )
    return episode


def test_a2c_integration() -> None:
    agent, learner = _build_a2c_pair()
    before = {name: value.detach().clone() for name, value in learner.agent_net.state_dict().items()}

    result = learner.update([
        _build_episode(1, terminated_at_end=True),
        _build_episode(2, terminated_at_end=False),
    ])

    assert learner.agent_net is learner.models["agent_net"]
    assert result["update_cnt"] == 1
    assert "pg_loss" in result
    assert "value_loss" in result
    assert "train/policy_loss" not in result
    assert "train/value_loss" not in result
    assert any(
        not torch.equal(before[name], value)
        for name, value in learner.agent_net.state_dict().items()
    ), "A2C update should modify policy parameters."
    snapshots = snapshot_model_weights(learner.models)
    assert set(snapshots) == {"agent_net"}
    assert all(tensor.device.type == "cpu" for tensor in snapshots["agent_net"].values())

    token_config = copy.deepcopy(learner.config.algo_config)
    learner.dynamic_modify_config({"lr": 1e-3, "use_rms_prop": False})
    assert learner.a2c.lr_schedule(1.0) == 1e-3
    assert isinstance(learner.a2c.policy.optimizer, torch.optim.Adam)

    with tempfile.TemporaryDirectory(prefix="dodorl-a2c-") as model_dir:
        model_files = learner.save_models(
            model_dir,
            {"agent_net": f"self_test_{learner.name}_agent_net_{learner.generation}.pth"},
        )
        assert len(model_files) == 1

    # Learner 发布的 ActorCriticPolicy 可直接同步到执行侧 A2C Agent。
    agent.reset_net(learner.agent_net)
    actions, hidden = agent.get_actions(
        obs={"solo-0": np.asarray([0.0, 1.0, 0.0], dtype=np.float32)},
        agent_list=["solo-0"],
        hidden_in={"solo-0": None},
        action_mask={"solo-0": [1, 1]},
        epsilon=0.0,
        evaluate=True,
    )
    assert actions["solo-0"].action_idx in (0, 1)
    assert hidden == {"solo-0": None}


if __name__ == "__main__":
    test_a2c_integration()
    print("A2C integration test passed.")
