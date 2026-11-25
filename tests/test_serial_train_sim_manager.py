"""SerialTrainRole的Simulation配置来源约束。"""
from __future__ import annotations

import pytest

from config import EnvConfig
from types import SimpleNamespace
from roles.serial_train import SerialTrainRole


def test_serial_train_accepts_local_sim_without_manager() -> None:
    env_config = EnvConfig(use_sim=True)
    sim_config = SimpleNamespace(config_source="local")

    SerialTrainRole._assert_supported(env_config, sim_config, None)


def test_serial_train_requires_enabled_manager_for_manager_source() -> None:
    env_config = EnvConfig(use_sim=True)
    sim_config = SimpleNamespace(config_source="manager")

    with pytest.raises(AssertionError, match="enabled SimManagerConfig"):
        SerialTrainRole._assert_supported(env_config, sim_config, None)
    with pytest.raises(AssertionError, match="enabled SimManagerConfig"):
        SerialTrainRole._assert_supported(
            env_config,
            sim_config,
            SimpleNamespace(use_sim=False),
        )

    SerialTrainRole._assert_supported(
        env_config,
        sim_config,
        SimpleNamespace(use_sim=True),
    )
