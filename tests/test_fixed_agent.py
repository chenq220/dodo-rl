from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from config import AgentConfig, AgentType
from envs.action.base import ActionTable
from envs.action.model import StepActionDefined
from envs.sense.aec_examples import CounterAECEnvironment
from envs.sense.paralle_examples import CounterParallelEnvironment
from scheduler.model import AgentCompose, ActionParam, EpisodeInfo
from scheduler.rollout_sample import RolloutSampleWorker
from utils import module_utils
from utils.toml_config import TomlTrainingConfig


def test_build_policy_agents_loads_configured_model_file(monkeypatch, tmp_path: Path) -> None:
    model_file = tmp_path / "fixed_agent.pth"
    model_file.touch()
    config = AgentConfig(
        name="fixed_agent",
        agent_type=AgentType.SB3.value,
        agent_names=["fixed_agent"],
        model_file_path=str(model_file),
        fixed_mode=True,
    )
    calls = []

    def fake_create_agent(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name=kwargs["agent_config"].name)

    monkeypatch.setattr(module_utils, "create_agent", fake_create_agent)

    agents = module_utils.build_policy_agents([config])

    assert [agent.name for agent in agents] == ["fixed_agent"]
    assert calls[0]["agent_net_source"] == "file"
    assert calls[0]["model_file_path"] == str(model_file)


def test_toml_allows_network_agent_without_learner_when_model_is_configured(
        tmp_path: Path,
) -> None:
    model_file = tmp_path / "fixed_agent.pth"
    config_file = tmp_path / "fixed_agent.toml"
    config_file.write_text(
        f'''
[agents.0]
name = "fixed_agent"
agent_type = "sb3"
agent_names = ["fixed_agent"]
model_file_path = "{model_file}"
fixed_mode = true

[agents.1]
name = "learning_agent"
agent_type = "sb3"
agent_names = ["learning_agent"]

[learners.0]
name = "learning_agent"
algo_type = "ppo"
''',
        encoding="utf-8",
    )

    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    fixed_agent = config.agents_config[0]
    assert fixed_agent.model_file_path == str(model_file)
    assert fixed_agent.fixed_mode is True


def test_toml_rejects_network_agent_without_learner_or_model(tmp_path: Path) -> None:
    config_file = tmp_path / "missing_fixed_model.toml"
    config_file.write_text(
        '''
[agents.0]
name = "fixed_agent"
agent_type = "sb3"
agent_names = ["fixed_agent"]
''',
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="model_file_path"):
        TomlTrainingConfig(config_file, project_root=tmp_path)


def test_sync_agents_from_model_manager_skips_fixed_and_non_learner_agents(monkeypatch):
    manager = MagicMock()
    manager.get_model.return_value = {"agent_net": {"weight": 1}}
    monkeypatch.setattr(module_utils, "extract_state_dict", lambda checkpoint: checkpoint)

    def _agent(name: str, *, fixed_mode: bool):
        net = MagicMock()
        net.to.return_value = net
        return SimpleNamespace(
            name=name,
            config=SimpleNamespace(fixed_mode=fixed_mode),
            device="cpu",
            agent_net=net,
        )

    learning = _agent("uav_team", fixed_mode=False)
    fixed = _agent("dynamic_threats", fixed_mode=True)
    extra = _agent("scripted", fixed_mode=False)

    module_utils.sync_agents_from_model_manager(
        [learning, fixed, extra],
        manager,
        learner_names=["uav_team"],
    )

    manager.get_model.assert_called_once_with(model_name="uav_team", output_type="state_dict")
    learning.agent_net.load_state_dict.assert_called_once()
    fixed.agent_net.load_state_dict.assert_not_called()
    extra.agent_net.load_state_dict.assert_not_called()


def test_records_replay_transition_skips_fixed_mode() -> None:
    assert RolloutSampleWorker._records_replay_transition(
        SimpleNamespace(config=SimpleNamespace(fixed_mode=True))
    ) is False
    assert RolloutSampleWorker._records_replay_transition(
        SimpleNamespace(config=SimpleNamespace(fixed_mode=False))
    ) is True
    assert RolloutSampleWorker._records_replay_transition(
        SimpleNamespace(config=SimpleNamespace())
    ) is True


class _StubPolicyAgent:
    def __init__(self, name, agent_ids, *, fixed_mode=False):
        self.name = name
        self.agent_list = list(agent_ids)
        self.agent_type = "rnn"
        self.device = "cpu"
        self.config = SimpleNamespace(agent_output_type="q", fixed_mode=fixed_mode)
        self.dim_info = SimpleNamespace(hidden_dim=1)
        self.action_table = ActionTable([
            StepActionDefined(action_key="counter", parameters=[], add_target=False)
        ])

    def get_init_hidden_state(self):
        return {agent_id: torch.zeros(1, 1) for agent_id in self.agent_list}

    def get_actions(self, agent_list, hidden_in, **kwargs):
        action = self.action_table.action_map[0]
        return (
            {agent_id: action for agent_id in agent_list},
            {agent_id: hidden_in[agent_id] for agent_id in agent_list},
        )


class _RecordingReplay:
    def __init__(self):
        self.items = []

    def push(self, item, **kwargs):
        self.items.append(item)


def _fixed_and_learning_agents():
    learning = _StubPolicyAgent("uav_team", ["agent_0"])
    fixed = _StubPolicyAgent("dynamic_threats", ["agent_1"], fixed_mode=True)
    return learning, fixed


def _fixed_replay_episode_info() -> EpisodeInfo:
    return EpisodeInfo(
        episode_num=1,
        task_id="fixed-replay",
        epsilon=0.0,
        deterministic=True,
        temperature=1.0,
        actions_param={
            "uav_team": ActionParam(epsilon=0.0, deterministic=True, temperature=1.0),
            "dynamic_threats": ActionParam(epsilon=0.0, deterministic=True, temperature=1.0),
        },
        seed=7,
        start_datetime=None,
        end_datetime=None,
        metadata={},
    )


def _assert_fixed_agent_not_pushed(transitions, replay: _RecordingReplay) -> None:
    assert list(transitions) == ["uav_team"]
    assert "dynamic_threats" not in transitions
    assert len(replay.items) == 1
    assert list(replay.items[0]) == ["uav_team"]


def test_parallel_rollout_does_not_record_fixed_mode_agent() -> None:
    learning, fixed = _fixed_and_learning_agents()
    replay = _RecordingReplay()
    worker = RolloutSampleWorker(
        env=CounterParallelEnvironment(["agent_0", "agent_1"], max_steps=2),
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=replay,
        agent_compose=AgentCompose(step_agent=[learning, fixed]),
        policy_agents=[learning, fixed],
    )
    transitions = worker.rolling_parallel_episode(
        _fixed_replay_episode_info(),
        random_sample=True,
        step_limit=1,
    )
    _assert_fixed_agent_not_pushed(transitions, replay)


def test_aec_rollout_does_not_record_fixed_mode_agent() -> None:
    learning, fixed = _fixed_and_learning_agents()
    replay = _RecordingReplay()
    worker = RolloutSampleWorker(
        env=CounterAECEnvironment(["agent_0", "agent_1"], max_steps=2),
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=replay,
        agent_compose=AgentCompose(step_agent=[learning, fixed]),
        policy_agents=[learning, fixed],
    )
    transitions = worker.rolling_aec_episode(
        _fixed_replay_episode_info(),
        random_sample=True,
        step_limit=2,
    )
    _assert_fixed_agent_not_pushed(transitions, replay)
