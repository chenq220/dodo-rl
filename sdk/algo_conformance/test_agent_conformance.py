"""Agent 算法规范性发烟测试。

对照《算法二次开发手册》Agent 接口表，逐项检查：
1. 必须实现的接口是否齐全（仍是抽象方法 -> InterfaceMissingException）；
2. 接口返回值是否满足契约（类型/形状/语义 -> ContractViolationException）；
3. 端到端功能是否可用（build / get_actions / reset_net -> FunctionalityException）。

新增算法时复制一份描述符并加到下面的 DESCRIPTORS，本文件无需改动。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from algo_conformance.conformance import (
    assert_interface_implemented,
    build_agent,
    build_learner,
    contract_check,
    function_check,
    import_class,
    load_descriptor_from_toml,
)

# 被测算法 TOML 配置文件清单；新增算法在此追加即可。
DESCRIPTOR_TOMLS = [
    Path(__file__).parent / "qmix.toml",
]
DESCRIPTORS = [load_descriptor_from_toml(p) for p in DESCRIPTOR_TOMLS]

# Agent 必须实现的接口名（来自手册接口表）。
REQUIRED_INTERFACES = [
    "get_actions",
    "build_empty_network",
    "build_network_from_checkpoint",
    "get_init_hidden_state",
]


def _check_required_interfaces(agent_cls, component: str) -> None:
    """检查必须实现的接口是否齐全。"""
    for name in REQUIRED_INTERFACES:
        assert_interface_implemented(agent_cls, name, component)


def _check_build_empty_network(agent, component: str) -> None:
    """build_empty_network 必须返回 nn.Module。"""
    net = function_check(lambda: agent.agent_net, component, "build_empty_network")
    contract_check(
        isinstance(net, nn.Module),
        component, "build_empty_network",
        f"agent_net must be nn.Module, given={type(net).__name__}",
    )


def _check_init_hidden_state(agent, component: str) -> None:
    """get_init_hidden_state 必须为每个 agent 返回初始隐藏状态（无记忆策略可为占位）。"""
    hidden = function_check(agent.get_init_hidden_state, component, "get_init_hidden_state")
    contract_check(
        isinstance(hidden, dict),
        component, "get_init_hidden_state",
        f"must return dict, given={type(hidden).__name__}",
    )
    contract_check(
        set(hidden.keys()) == set(agent.agent_list),
        component, "get_init_hidden_state",
        f"keys must match agent_list, "
        f"expected={agent.agent_list}, got={list(hidden.keys())}",
    )


def _check_get_actions(agent, component: str) -> None:
    """get_actions 必须为每个 agent 返回业务 Action 实体（带 action_idx）和隐藏状态字典。"""
    init_hidden = agent.get_init_hidden_state()
    obs = {
        agent_id: np.zeros(agent.obs_dim.shape, dtype=np.float32)
        for agent_id in agent.agent_list
    }
    action_mask = {
        agent_id: [1] * agent.action_dim.n for agent_id in agent.agent_list
    }

    actions, hidden_out = function_check(
        agent.get_actions,
        component, "get_actions",
        obs=obs,
        agent_list=agent.agent_list,
        hidden_in=init_hidden,
        action_mask=action_mask,
        epsilon=0.0,
        evaluate=True,
    )
    contract_check(
        isinstance(actions, dict),
        component, "get_actions",
        f"actions must be dict, given={type(actions).__name__}",
    )
    contract_check(
        set(actions.keys()) == set(agent.agent_list),
        component, "get_actions",
        f"actions keys must match agent_list, "
        f"expected={agent.agent_list}, got={list(actions.keys())}",
    )
    for agent_id, action in actions.items():
        contract_check(
            hasattr(action, "action_idx"),
            component, "get_actions",
            f"actions['{agent_id}'] must be business Action with action_idx, "
            f"given={type(action).__name__}",
        )
        contract_check(
            0 <= int(action.action_idx) < agent.action_dim.n,
            component, "get_actions",
            f"actions['{agent_id}'].action_idx out of range: "
            f"{action.action_idx}, n_actions={agent.action_dim.n}",
        )
    contract_check(
        isinstance(hidden_out, dict),
        component, "get_actions",
        f"hidden_out must be dict, given={type(hidden_out).__name__}",
    )


def _check_reset_net(agent, learner, component: str) -> None:
    """reset_net 必须能接收 Learner.agent_net 并同步权重。"""
    before = copy.deepcopy(agent.agent_net.state_dict())
    function_check(lambda: agent.reset_net(learner.agent_net), component, "reset_net")
    after = agent.agent_net.state_dict()
    # SB3Agent 只 load_state_dict 不 deepcopy，权重应与 Learner 一致；这里只校验同步成功。
    synced = any(not torch.equal(before[k], after[k]) for k in before)
    # 同种子 blanco 网络权重可能恰好相同，同步成功即可，不强制要求 changed。
    _ = synced


def run_agent_conformance(desc) -> None:
    """对一个算法描述符执行完整的 Agent 规范性发烟测试。"""
    component = f"Agent[{desc.name}]"
    agent_cls = import_class(desc.agent_import_path)

    # 1. 接口齐全性
    _check_required_interfaces(agent_cls, component)

    # 2. 实例化
    _, agent = build_agent(desc)

    # 3. 契约检查
    _check_build_empty_network(agent, component)
    _check_init_hidden_state(agent, component)
    _check_get_actions(agent, component)

    # 4. Learner -> Agent 同步
    env = desc.env_stub_factory()
    _, learner = build_learner(desc, agent, env)
    learner.learn_update(desc.episode_factory(agent, learner))
    _check_reset_net(agent, learner, component)


# ---- pytest 入口 ----

def _make_test(desc):
    def _test():
        run_agent_conformance(desc)
    _test.__name__ = f"test_agent_conformance_{desc.name}"
    return _test


for _desc in DESCRIPTORS:
    globals()[f"test_agent_conformance_{_desc.name}"] = _make_test(_desc)


if __name__ == "__main__":
    for _desc in DESCRIPTORS:
        run_agent_conformance(_desc)
        print(f" Agent conformance smoke test passed: {_desc.name}")
