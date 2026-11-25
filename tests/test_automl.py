"""AutoML 模块：单测覆盖 advice 应答 + restart 归档（假资源，测完清理）。"""
from __future__ import annotations

import os.path
import sys
import uuid
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AutoMLConfig, AutoMLOptunaConfig, LearnerConfig, StorageType, TrainConfig
from scheduler.automl import AutoML
from scheduler.model import (
    AutoMLEnvMetadataSearchParamConfig,
    AutoMLLearnerSearchParamConfig,
    AutoMLSeedSearchConfig,
    TrainingRunArchive,
    TrainingSnapshot,
)
from scheduler.utils.automl_config import finalize_auto_ml_config
from scheduler.utils.automl_core import resolve_optuna_sqlite_storage


def _snapshot(episode: int, metric: float) -> TrainingSnapshot:
    rewards = [metric] * episode
    return TrainingSnapshot(
        task_id="automl_smoke",
        stage_name="combat",
        generation=0,
        completed_episodes=episode,
        max_episodes=200,
        reward_curve={"archer": rewards},
        learn_metrics={},
        current_algo_configs={},
        tunable_catalog={},
    )


def test_automl_stage_success_rate_uses_episode_window() -> None:
    from scheduler.utils.automl_core import aggregate_reward

    snapshot = _snapshot(4, 99.0)
    snapshot.episode_metrics = {"stage_success": [0.0, 1.0, 1.0, 0.0]}
    config = AutoMLConfig(
        subject_name="success_rate",
        metric_primary="stage_success_rate",
        reward_window=3,
    )

    assert aggregate_reward(snapshot, config) == pytest.approx(2.0 / 3.0)


def test_automl_advice_and_archive(tmp_path: Path) -> None:
    subject_name = f"smoke_{uuid.uuid4().hex[:8]}"
    archive_root = tmp_path / "archives"
    study_db = Path(
        resolve_optuna_sqlite_storage(subject_name, "combat")[1].removeprefix("sqlite:///")
    )

    automl = AutoML(
        finalize_auto_ml_config(AutoMLConfig(
            subject_name=subject_name,
            monitor_enabled=True,
            archive_enabled=True,
            collapse_restart_ratio=0.35,
            collapse_min_episodes=30,
            reward_window=20,
            storage_type=StorageType.LOCAL.value,
            archive_root_dir=str(archive_root),
            optuna=AutoMLOptunaConfig(load_if_exists=False, pruner_name="none"),
            learner_search_space=[AutoMLLearnerSearchParamConfig(
                agent_name="archer", path="algo_config.lr",
            )],
            env_metadata_search_space=[AutoMLEnvMetadataSearchParamConfig(
                path="reward_weights.navigation", type="float", low=0.1, high=0.5,
            )],
        )),
        [LearnerConfig(name="archer", algo_type="mappo")],
        stage_name="combat",
    )

    try:
        assert automl._trial is None
        # 1) 请求建议并应答：continue → restart（坍塌）
        assert automl.advice(_snapshot(40, 5.0)).decision == "continue"
        assert automl._trial is None
        restart = automl.advice(_snapshot(80, 1.0))
        assert restart.decision == "restart"
        assert automl._trial is not None
        assert restart.algo_config_overrides["archer"]["lr"] is not None

        # 2) restart 归档：假资源写入后移动进归档包
        fake_reward = tmp_path / "reward.png"
        fake_ckpt = tmp_path / "checkpoint.pth"
        fake_reward.write_bytes(b"png")
        fake_ckpt.write_bytes(b"weights")

        archive_dir = Path(automl.archive_train(TrainingRunArchive(
            env_name="demo_env",
            task_id="automl_smoke",
            stage_name="combat",
            generation=0,
            archive_seq=0,
            completed_episodes=80,
            episode_rows=[{"episode": 80, "generation": 0, "archer_reward": 1.0}],
            metric_output_rows={
                "team_kills": [{"episode": 80, "generation": 0, "value": 3.0}],
            },
            artifact_paths={
                "reward_curve": [str(fake_reward)],
                "models": [str(fake_ckpt)],
            },
            snapshot=_snapshot(80, 1.0),
            advice=restart,
            run_seed=1215682775,
            initialization_seed=1472572158,
        )))

        assert archive_dir.is_dir()
        assert (archive_dir / "algos_metrics.csv").exists()
        assert (archive_dir / "team_kills_metrics.csv").exists()
        assert not (archive_dir / "business_metrics.csv").exists()
        assert not (archive_dir / "combat_score.csv").exists()
        with open(archive_dir / "advice.yaml", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp)
        assert payload["overview"]["decision"] == "restart"
        assert payload["overview"]["run_seed"] == 1215682775
        assert payload["overview"]["initialization_seed"] == 1472572158
        assert payload["advice"]["type"] == "next_trial"
        assert payload["optuna"]["current_trial"]["state"] == "BASELINE"
        assert payload["optuna"]["next_trial"]["number"] == automl._trial.trial_number
        assert payload["advice"]["toml_overrides"]["learner_config"]["archer"]["algo_config"]["lr"]
        assert (archive_dir / "artifacts" / "reward_curve" / "reward.png").exists()
        assert (archive_dir / "artifacts" / "models" / "checkpoint.pth").exists()
        assert not fake_reward.exists()
        assert not fake_ckpt.exists()
    finally:
        if study_db.is_file():
            study_db.unlink()


def test_automl_defers_seed_ask_until_baseline_needs_restart(tmp_path: Path) -> None:
    subject_name = f"deferred_seed_{uuid.uuid4().hex[:8]}"
    study_db = Path(
        resolve_optuna_sqlite_storage(subject_name, "deploy")[1].removeprefix("sqlite:///")
    )
    fixed_seed = 852567296
    train_config = TrainConfig(seed=fixed_seed)
    automl = AutoML(
        finalize_auto_ml_config(AutoMLConfig(
            subject_name=subject_name,
            monitor_enabled=True,
            target_metric=1.0,
            target_metric_deadline_episodes=20,
            archive_root_dir=str(tmp_path / "archives"),
            optuna=AutoMLOptunaConfig(load_if_exists=False, pruner_name="none"),
            seed_search=AutoMLSeedSearchConfig(),
        )),
        [LearnerConfig(name="knight", algo_type="mappo")],
        stage_name="deploy",
    )
    try:
        # AutoML 初始化和观察都不得改写当前配置，也不得提前占用 trial。
        assert automl._trial is None
        assert train_config.seed == fixed_seed
        assert automl.advice(_snapshot(10, 2.0)).decision == "continue"
        assert automl._trial is None
        assert train_config.seed == fixed_seed

        # 只有当前运行在截止点未达标时，才 ask 下一次运行使用的 seed。
        restart = automl.advice(_snapshot(20, 0.0))
        assert restart.decision == "restart"
        assert restart.seed is not None
        assert automl._trial is not None
        assert train_config.seed == fixed_seed
    finally:
        if study_db.is_file():
            study_db.unlink()


def test_automl_prunes_when_target_metric_is_missed(tmp_path: Path) -> None:
    subject_name = f"target_{uuid.uuid4().hex[:8]}"
    study_db = Path(
        resolve_optuna_sqlite_storage(subject_name, "deploy")[1].removeprefix("sqlite:///")
    )
    automl = AutoML(
        finalize_auto_ml_config(AutoMLConfig(
            subject_name=subject_name,
            monitor_enabled=True,
            archive_root_dir=str(tmp_path / "archives"),
            target_metric=3.0,
            target_metric_deadline_episodes=20,
            optuna=AutoMLOptunaConfig(load_if_exists=False, pruner_name="none"),
            learner_search_space=[AutoMLLearnerSearchParamConfig(
                agent_name="archer", path="algo_config.lr",
            )],
            env_metadata_search_space=[AutoMLEnvMetadataSearchParamConfig(
                path="reward_weights.navigation", type="float", low=0.1, high=0.5,
            )],
        )),
        [LearnerConfig(name="archer", algo_type="mappo")],
        stage_name="deploy",
    )
    try:
        advice = automl.advice(_snapshot(20, 0.06))
        assert advice.decision == "restart"
        assert advice.reason.startswith("target_metric_missed:")
        assert 0.1 <= advice.env_metadata_overrides["reward_weights"]["navigation"] <= 0.5
    finally:
        if study_db.is_file():
            study_db.unlink()
