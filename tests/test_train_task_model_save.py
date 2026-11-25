"""TrainTaskManager 模型落盘规范：文件名、optimo 覆盖、pruned 归档、final 触发。"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, PpoConfig, TrainConfig
from scheduler.model import AutoMLAdvice
from scheduler.train_task import TrainTaskManager, format_model_filename


class _FakeLearner:
    def __init__(self, name: str, generation: int = 0):
        self.name = name
        self.generation = generation
        self.models = {"agent_net": object()}
        self.save_calls: list[tuple[str, dict[str, str]]] = []

    def save_models(self, save_model_path: str, save_model_name: dict[str, str]) -> list[str]:
        os.makedirs(save_model_path, exist_ok=True)
        self.save_calls.append((save_model_path, dict(save_model_name)))
        saved: list[str] = []
        for model_name, filename in save_model_name.items():
            file_path = os.path.join(save_model_path, filename)
            with open(file_path, "w", encoding="utf-8") as fp:
                fp.write(f"{self.name}:{model_name}")
            saved.append(file_path)
        return saved


def _agent_config(name: str) -> AgentConfig:
    return AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name])


def _learner_config(name: str) -> LearnerConfig:
    learner_config = LearnerConfig(name=name, algo_type=AlgorithmType.PPO.value)
    assert isinstance(learner_config.algo_config, PpoConfig)
    learner_config.algo_config.learn_times_per_episode = 1
    return learner_config


def _make_caller(tm: TrainTaskManager, learners: list[_FakeLearner], save_dir: str):
    def caller(name_prefix: str) -> list[str]:
        paths: list[str] = []
        for learner in learners:
            filenames = {
                model_name: tm.resolve_model_filename(name_prefix, learner.name, model_name)
                for model_name in learner.models
            }
            paths.extend(learner.save_models(save_dir, filenames))
        return paths

    return caller


def test_format_model_filename_pattern() -> None:
    filename = format_model_filename(
        "final",
        "knight",
        "combat",
        "agent_net",
        2,
        "task-01/demo",
    )
    assert filename == "final_task-01_demo_combat_2_knight_agent_net.pth"


def test_resolve_model_filename_matches_tm_fields(tmp_path: Path) -> None:
    tm = TrainTaskManager(
        config=TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("archer")],
        learner_configs=[_learner_config("archer")],
        stage_name="mid",
    )
    try:
        expected = format_model_filename(
            "optimo", "archer", "mid", "agent_net", tm.generation, tm.task_id,
        )
        assert tm.resolve_model_filename("optimo", "archer", "agent_net") == expected
    finally:
        tm.close()


def test_optimo_overwrites_same_generation(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "models")
    learner = _FakeLearner("knight")
    tm = TrainTaskManager(
        config=TrainConfig(
            max_episode=4,
            sample_learn_max_lag=None,
            model_save_dir=save_dir,
            save_optimo_models=True,
        ),
        policy_agent_configs=[_agent_config("knight")],
        learner_configs=[_learner_config("knight")],
    )
    tm.register_learner_save_caller(_make_caller(tm, [learner], save_dir))
    try:
        for episode_num, reward in ((1, 1.0), (2, 0.5), (3, 2.0), (4, 2.0)):
            token = tm.get_sample_task_info()
            assert token is not None and token.episode_num == episode_num
            tm.report_sample_result(
                episode_num, total_rewards={"knight": reward}, lease_id=token.lease_id,
            )

        optimo_name = tm.resolve_model_filename("optimo", "knight", "agent_net")
        optimo_path = os.path.join(save_dir, optimo_name)
        optimo_calls = [
            names for _, names in learner.save_calls
            if list(names.values())[0].startswith("optimo_")
        ]
        assert len(optimo_calls) == 3
        with open(optimo_path, encoding="utf-8") as fp:
            assert fp.read() == "knight:agent_net"
    finally:
        tm.close()


def test_prepare_archive_saves_pruned(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "models")
    learner = _FakeLearner("knight")
    auto_ml = MagicMock()
    auto_ml.config.archive_enabled = True
    tm = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            model_save_dir=save_dir,
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        policy_agent_configs=[_agent_config("knight")],
        learner_configs=[_learner_config("knight")],
        auto_ml=auto_ml,
    )
    tm.register_learner_save_caller(_make_caller(tm, [learner], save_dir))
    advice = AutoMLAdvice(decision="restart", reason="test_prune")
    try:
        tm._prepare_archive_resources(advice, episode_num=1, generation=0)
        pruned_name = tm.resolve_model_filename("pruned", "knight", "agent_net")
        pruned_path = os.path.join(save_dir, pruned_name)
        assert os.path.isfile(pruned_path)
        assert pruned_path in tm._checkpoint_model_paths
    finally:
        tm.close()


def test_notify_stage_complete_saves_final(tmp_path: Path) -> None:
    save_dir = str(tmp_path / "models")
    learner = _FakeLearner("archer")
    tm = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            model_save_dir=save_dir,
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        policy_agent_configs=[_agent_config("archer")],
        learner_configs=[_learner_config("archer")],
    )
    tm.register_learner_save_caller(_make_caller(tm, [learner], save_dir))
    try:
        archived = tm.notify_stage_complete()
        final_name = tm.resolve_model_filename("final", "archer", "agent_net")
        assert archived is not None
        assert os.path.isfile(os.path.join(archived.archive_dir, "models", "train", final_name))
        assert not os.path.exists(os.path.join(save_dir, final_name))
        assert any(name.startswith("final_") for _, names in learner.save_calls for name in names.values())
    finally:
        tm.close()


if __name__ == "__main__":
    import tempfile

    test_format_model_filename_pattern()
    test_resolve_model_filename_matches_tm_fields(Path(tempfile.mkdtemp()))
    test_optimo_overwrites_same_generation(Path(tempfile.mkdtemp()))
    test_prepare_archive_saves_pruned(Path(tempfile.mkdtemp()))
    print("all model save tests passed")
