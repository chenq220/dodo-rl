"""算法规范性发烟测试脚手架。

合并三类工具：
- AlgoDescriptor：发烟测试配置项数据结构 + import_class 导入工具；
- 检查工具：抽象方法探测、契约断言包装、功能调用包装、按描述符实例化；
- TOML 加载：复用 TomlTrainingConfig 解析配置 + 通用环境桩 / episode 工厂。

新增算法只需写一份 TOML（格式同训练 TOML + [conformance] 小节），本模块不用改。
"""
from __future__ import annotations

import importlib
import inspect
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import gymnasium as gym
import numpy as np

from common.creator import create_agent, create_learner
from common.exception import (
    ContractViolationException,
    FunctionalityException,
    InterfaceMissingException,
)
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from utils.toml_config import TomlTrainingConfig


# ============================================================
# 1. 描述符数据结构
# ============================================================

@dataclass
class AlgoDescriptor:
    """单个算法的发烟测试配置项。"""

    name: str
    learner_import_path: str
    agent_import_path: str
    learner_config: Any
    agent_config: Any
    env_stub_factory: Callable[[], Any]
    episode_factory: Callable[[Any, Any], list]


def import_class(path: str):
    """按 "module.path.ClassName" 导入类。"""
    module_path, _, class_name = path.rpartition(".")
    if not module_path:
        raise ValueError(f"invalid import path: {path}")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ============================================================
# 2. 检查工具
# ============================================================

def _is_abstract_on(cls: type, attr_name: str) -> bool:
    """判断 cls 上的属性是否仍是未实现的抽象方法。"""
    for klass in inspect.getmro(cls):
        if attr_name in klass.__dict__:
            return bool(getattr(klass.__dict__[attr_name], "__isabstractmethod__", False))
    return False


def assert_interface_implemented(cls: type, attr_name: str, component: str) -> None:
    """断言子类已实现指定接口；仍是抽象方法则抛 InterfaceMissingException。"""
    if _is_abstract_on(cls, attr_name):
        raise InterfaceMissingException(
            component=component,
            contract=attr_name,
            detail=f"{cls.__name__} did not implement required interface '{attr_name}'",
        )


def build_agent(desc: AlgoDescriptor):
    """按描述符实例化 Agent。"""
    agent_cls = import_class(desc.agent_import_path)
    return agent_cls, create_agent(
        agent_type=desc.agent_config.agent_type,
        agent_config=desc.agent_config,
        agent_net_source="blanco",
        agent_list=desc.agent_config.agent_names,
        device=desc.agent_config.device,
    )


def build_learner(desc: AlgoDescriptor, agent, env):
    """按描述符实例化 Learner。"""
    learner_cls = import_class(desc.learner_import_path)
    return learner_cls, create_learner(
        learner_config=desc.learner_config,
        agent=agent,
        env=env,
        device=desc.learner_config.device,
        net_source="blanco",
    )


def contract_check(condition: bool, component: str, contract: str, detail: str) -> None:
    """契约断言：condition 为 False 时抛 ContractViolationException。"""
    if not condition:
        raise ContractViolationException(
            component=component, contract=contract, detail=detail,
        )


def function_check(func: Callable[..., Any], component: str, contract: str, *args, **kwargs) -> Any:
    """功能调用包装：执行失败时抛 FunctionalityException。"""
    try:
        return func(*args, **kwargs)
    except FunctionalityException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise FunctionalityException(
            component=component, contract=contract,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


# ============================================================
# 3. TOML 加载 + 通用环境桩 / episode 工厂
# ============================================================

def load_descriptor_from_toml(config_path: str | Path) -> AlgoDescriptor:
    """解析 conformance TOML，返回可直接用于发烟测试的 AlgoDescriptor。

    复用 TomlTrainingConfig 解析 [agents.N] / [learners.N] / [env_config]，
    额外读取 [conformance] 小节获取发烟测试专用字段。
    """
    config_path = Path(config_path)

    # 1. 用 TomlTrainingConfig 解析 Agent/Learner/Env 配置。
    training_config = TomlTrainingConfig(config_path)
    agent_config = training_config.agents_config[0]
    learner_config = training_config.learners_config[0]
    env_config = training_config.env_config

    # 2. 读 [conformance] 小节获取测试专用字段。
    with config_path.open("rb") as fp:
        raw = tomllib.load(fp)
    conf = raw["conformance"]

    learner_import_path = conf["learner_import_path"]
    agent_import_path = conf["agent_import_path"]
    sample_type = conf.get("sample_type", "multi_agent")
    episode_steps = int(conf["episode_steps"])
    episode_count = int(conf["episode_count"])

    # 3. 从已解析的配置中提取 episode 工厂所需维度。
    agent_names: List[str] = list(agent_config.agent_names)
    obs_dim = agent_config.obs_dim
    state_dim = env_config.state_dim
    n_actions = len(agent_config.define_action[0].parameters[0].values)
    max_step_per_episode = env_config.max_step_per_episode

    # 4. 构建通用环境桩和 episode 工厂。
    env_stub_factory = _make_env_stub_factory(state_dim, max_step_per_episode)
    episode_factory = _make_episode_factory(
        sample_type, agent_names, obs_dim, state_dim, n_actions,
        episode_steps, episode_count,
    )

    return AlgoDescriptor(
        name=learner_config.algo_type,
        learner_import_path=learner_import_path,
        agent_import_path=agent_import_path,
        learner_config=learner_config,
        agent_config=agent_config,
        env_stub_factory=env_stub_factory,
        episode_factory=episode_factory,
    )


def _make_env_stub_factory(state_dim: int, max_step_per_episode: int) -> Callable:
    class _StubEnv:
        def __init__(self) -> None:
            self.config = SimpleNamespace(max_step_per_episode=max_step_per_episode)
            self._state_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32,
            )

        def state_space(self) -> gym.spaces.Space:
            return self._state_space

    return _StubEnv


def _make_episode_factory(
    sample_type: str,
    agent_names: List[str],
    obs_dim: int,
    state_dim: int,
    n_actions: int,
    episode_steps: int,
    episode_count: int,
) -> Callable[[Any, Any], list]:
    if sample_type == "multi_agent":
        return _MultiAgentEpisodeFactory(
            agent_names, obs_dim, state_dim, n_actions, episode_steps, episode_count,
        )
    if sample_type == "single_agent":
        return _SingleAgentEpisodeFactory(
            agent_names[0], obs_dim, n_actions, episode_steps, episode_count,
        )
    raise ValueError(f"unsupported sample_type in conformance toml: {sample_type}")


class _MultiAgentEpisodeFactory:
    def __init__(self, agent_names, obs_dim, state_dim, n_actions, steps, count):
        self._agent_names = agent_names
        self._obs_dim = obs_dim
        self._state_dim = state_dim
        self._n_actions = n_actions
        self._steps = steps
        self._count = count

    def __call__(self, agent: Any, learner: Any) -> list:
        return [self._build_episode(i + 1) for i in range(self._count)]

    def _build_episode(self, episode_num: int) -> EpisodeTransition:
        episode = EpisodeTransition()
        episode.episode_num = episode_num
        for step in range(self._steps):
            obs = {
                aid: np.asarray(
                    [episode_num, step, i, (step + i) % 2], dtype=np.float32
                )[: self._obs_dim]
                for i, aid in enumerate(self._agent_names)
            }
            next_obs = {
                aid: np.asarray(
                    [episode_num, step + 1, i, (step + 1 + i) % 2], dtype=np.float32
                )[: self._obs_dim]
                for i, aid in enumerate(self._agent_names)
            }
            action = {
                aid: type("A", (), {"action_idx": (step + i) % self._n_actions})()
                for i, aid in enumerate(self._agent_names)
            }
            reward = {
                aid: 1.0 if (step + i) % 2 == 0 else -0.25
                for i, aid in enumerate(self._agent_names)
            }
            state = np.zeros(self._state_dim, dtype=np.float32)
            state[0] = episode_num
            state[1] = step
            next_state = np.zeros(self._state_dim, dtype=np.float32)
            next_state[0] = episode_num
            next_state[1] = step + 1
            avail = {aid: [1] * self._n_actions for aid in self._agent_names}
            episode.add_step(
                obs=obs,
                action=action,
                last_action={aid: None for aid in self._agent_names},
                reward=reward,
                next_obs=next_obs,
                termination={aid: (step == self._steps - 1) for aid in self._agent_names},
                truncation={aid: False for aid in self._agent_names},
                state=state,
                next_state=next_state,
                avail_actions_next=avail,
            )
        return episode


class _SingleAgentEpisodeFactory:
    def __init__(self, agent_name, obs_dim, n_actions, steps, count):
        self._agent_name = agent_name
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        self._steps = steps
        self._count = count

    def __call__(self, agent: Any, learner: Any) -> list:
        return [self._build_episode(i + 1) for i in range(self._count)]

    def _build_episode(self, episode_num: int) -> SAEpisodeTransition:
        episode = SAEpisodeTransition()
        episode.episode_num = episode_num
        for step in range(self._steps):
            obs = np.asarray(
                [episode_num, step, step % 2], dtype=np.float32
            )[: self._obs_dim]
            next_obs = np.asarray(
                [episode_num, step + 1, (step + 1) % 2], dtype=np.float32
            )[: self._obs_dim]
            episode.add_step(
                obs=obs,
                action=step % self._n_actions,
                reward=1.0 if step % 2 == 0 else -0.25,
                next_obs=next_obs,
                terminated=(step == self._steps - 1),
                truncated=False,
                avail_actions=[1] * self._n_actions,
            )
        return episode
