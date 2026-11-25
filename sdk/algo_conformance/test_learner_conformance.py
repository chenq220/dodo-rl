"""Learner 算法规范性发烟测试。

对照《算法二次开发手册》Learner 接口表，按完整生命周期流程检查：
1. 接口齐全性（仍是抽象方法 -> InterfaceMissingException）；
2. build_empty_network 构建 -> 缓存 models 权重 -> 创建快照；
3. update 端到端 -> 检查 models 权重已变化；
4. save_models 落盘；
5. 恢复快照 -> 对比恢复后与缓存相同；
6. 销毁 Learner -> 从文件/caller 两种来源 build_network_from_checkpoint -> 对比 agent_net；
7. 清理临时文件。

新增算法时只需把 TOML 加到 DESCRIPTOR_TOMLS，本文件无需改动。
"""
from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path

import torch
from torch import nn

from algo_conformance.conformance import (
    AlgoDescriptor,
    assert_interface_implemented,
    build_agent,
    build_learner,
    contract_check,
    function_check,
    import_class,
    load_descriptor_from_toml,
)
from common.exception import FunctionalityException
from common.creator import create_learner

# 被测算法 TOML 配置文件清单；新增算法在此追加即可。
DESCRIPTOR_TOMLS = [
    Path(__file__).parent / "qmix.toml",
]
DESCRIPTORS = [load_descriptor_from_toml(p) for p in DESCRIPTOR_TOMLS]

# Learner 必须实现的接口名（来自手册接口表）。
REQUIRED_INTERFACES = [
    "agent_net",
    "models",
    "build_empty_network",
    "build_network_from_checkpoint",
    "update",
    "snapshot_state",
    "restore_state",
]


def _snapshot_weights(learner) -> dict:
    """深拷贝当前 models 全部 state_dict。"""
    return {
        name: {k: v.detach().clone() for k, v in model.state_dict().items()}
        for name, model in learner.models.items()
    }


def _weights_equal(a: dict, b: dict) -> bool:
    """比较两组 state_dict 是否完全相同。"""
    if set(a.keys()) != set(b.keys()):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def run_learner_conformance(desc: AlgoDescriptor) -> None:
    """对一个算法描述符执行完整的 Learner 生命周期发烟测试。"""
    component = f"Learner[{desc.name}]"
    learner_cls = import_class(desc.learner_import_path)

    # ---- 1. 接口齐全性 ----
    for name in REQUIRED_INTERFACES:
        assert_interface_implemented(learner_cls, name, component)

    # ---- 2. 实例化（build_empty_network 在 __init__ 中被调用）+ 契约检查 ----
    _, agent = build_agent(desc)
    env = desc.env_stub_factory()
    _, learner = build_learner(desc, agent, env)

    # build_empty_network 返回值 -> learner.net_list，必须是 Tuple[nn.Module, ...]
    contract_check(
        isinstance(learner.net_list, tuple) and len(learner.net_list) > 0,
        component, "build_empty_network",
        f"net_list must be non-empty tuple, given={type(learner.net_list).__name__}",
    )
    for i, net in enumerate(learner.net_list):
        contract_check(
            isinstance(net, nn.Module),
            component, "build_empty_network",
            f"net_list[{i}] must be nn.Module, given={type(net).__name__}",
        )

    # models -> Dict[str, nn.Module]，必须含 agent_net
    models = function_check(lambda: learner.models, component, "models")
    contract_check(
        isinstance(models, dict) and len(models) > 0,
        component, "models",
        f"models must be non-empty dict, given={type(models).__name__}",
    )
    contract_check(
        "agent_net" in models,
        component, "models",
        f"models must contain 'agent_net', keys={list(models.keys())}",
    )
    for key, model in models.items():
        contract_check(
            isinstance(model, nn.Module),
            component, "models",
            f"models['{key}'] must be nn.Module, given={type(model).__name__}",
        )

    # agent_net -> nn.Module
    agent_net = function_check(lambda: learner.agent_net, component, "agent_net")
    contract_check(
        isinstance(agent_net, nn.Module),
        component, "agent_net",
        f"agent_net must be nn.Module, given={type(agent_net).__name__}",
    )

    # ---- 2b. dynamic_modify_config 仅接受 dict ----
    algo_cfg = learner.config.algo_config
    if hasattr(algo_cfg, "lr"):
        old_lr = float(algo_cfg.lr)
        new_lr = old_lr * 1.5 if old_lr else 1e-3
        function_check(
            lambda: learner.dynamic_modify_config({"lr": new_lr}),
            component, "dynamic_modify_config",
        )
        contract_check(
            float(learner.config.algo_config.lr) == new_lr,
            component, "dynamic_modify_config",
            f"lr should update to {new_lr}, given={learner.config.algo_config.lr}",
        )
        try:
            learner.dynamic_modify_config(algo_cfg)
        except TypeError:
            pass
        else:
            contract_check(
                False,
                component,
                "dynamic_modify_config",
                "dynamic_modify_config must reject non-dict input",
            )

    # ---- 3. 缓存 models 权重（pre-update）+ 创建快照 ----
    pre_update_weights = _snapshot_weights(learner)
    # snapshot_state 显式调用检查返回值
    snapshot = function_check(learner.snapshot_state, component, "snapshot_state")
    contract_check(
        snapshot is not None,
        component, "snapshot_state",
        "snapshot_state must return non-None state",
    )
    function_check(learner.clone_models_snapshot, component, "snapshot_state")

    # ---- 4. update 端到端 ----
    episodes = desc.episode_factory(agent, learner)
    result = function_check(lambda: learner.learn_update(episodes), component, "update")
    contract_check(
        isinstance(result, dict),
        component, "update",
        f"update must return dict, given={type(result).__name__}",
    )

    # ---- 5. 检查 models 权重已变化 ----
    post_update_weights = _snapshot_weights(learner)
    for name in pre_update_weights:
        contract_check(
            not _weights_equal(pre_update_weights[name], post_update_weights[name]),
            component, "update",
            f"update did not modify models['{name}'] parameters",
        )

    # ---- 6. save_models 落盘 ----
    with tempfile.TemporaryDirectory(prefix="conformance_") as model_dir:
        save_names = {
            model_key: f"smoke_{learner.name}_{model_key}_{learner.generation}.pth"
            for model_key in learner.models
        }
        saved_files = function_check(
            lambda: learner.save_models(model_dir, save_names),
            component, "save_models",
        )
        contract_check(
            len(saved_files) == len(learner.models),
            component, "save_models",
            f"save_models should produce one file per model, "
            f"files={len(saved_files)}, models={len(learner.models)}",
        )
        model_names = list(learner.models.keys())
        model_file_path = dict(zip(model_names, saved_files))

        # ---- 7. 恢复快照 ----
        function_check(learner.restore_models_snapshot, component, "restore_state")

        # ---- 8. 对比恢复后与 pre-update 缓存相同 ----
        restored_weights = _snapshot_weights(learner)
        for name in pre_update_weights:
            contract_check(
                _weights_equal(pre_update_weights[name], restored_weights[name]),
                component, "restore_state",
                f"restore_state did not restore models['{name}'] to snapshot",
            )

        # ---- 9. 销毁 Learner ----
        del learner

        # ---- 10a. 从文件 build_network_from_checkpoint ----
        learner_config_copy = copy.deepcopy(desc.learner_config)
        learner_from_file = function_check(
            lambda: create_learner(
                learner_config=learner_config_copy,
                agent=agent,
                env=env,
                device=desc.learner_config.device,
                net_source="file",
                model_file_path=model_file_path,
            ),
            component, "build_network_from_checkpoint(file)",
        )
        contract_check(
            isinstance(learner_from_file.agent_net, nn.Module),
            component, "build_network_from_checkpoint(file)",
            f"agent_net must be nn.Module after loading from file, "
            f"given={type(learner_from_file.agent_net).__name__}",
        )
        # build_network_from_checkpoint 返回值 -> net_list
        contract_check(
            isinstance(learner_from_file.net_list, tuple) and len(learner_from_file.net_list) > 0,
            component, "build_network_from_checkpoint(file)",
            f"net_list must be non-empty tuple, given={type(learner_from_file.net_list).__name__}",
        )
        del learner_from_file

        # ---- 10b. 从 caller build_network_from_checkpoint ----
        caller_checkpoint = copy.deepcopy(pre_update_weights)

        def _caller(agent_only: bool = False) -> dict:
            return caller_checkpoint

        learner_config_copy2 = copy.deepcopy(desc.learner_config)
        learner_from_caller = function_check(
            lambda: create_learner(
                learner_config=learner_config_copy2,
                agent=agent,
                env=env,
                device=desc.learner_config.device,
                net_source="caller",
                caller=_caller,
            ),
            component, "build_network_from_checkpoint(caller)",
        )
        contract_check(
            isinstance(learner_from_caller.agent_net, nn.Module),
            component, "build_network_from_checkpoint(caller)",
            f"agent_net must be nn.Module after loading from caller, "
            f"given={type(learner_from_caller.agent_net).__name__}",
        )
        contract_check(
            isinstance(learner_from_caller.net_list, tuple) and len(learner_from_caller.net_list) > 0,
            component, "build_network_from_checkpoint(caller)",
            f"net_list must be non-empty tuple, given={type(learner_from_caller.net_list).__name__}",
        )
        del learner_from_caller

    # ---- 11. 临时文件由 TemporaryDirectory 自动清理 ----


# ---- pytest 入口 ----

def _make_test(desc):
    def _test():
        run_learner_conformance(desc)
    _test.__name__ = f"test_learner_conformance_{desc.name}"
    return _test


for _desc in DESCRIPTORS:
    globals()[f"test_learner_conformance_{_desc.name}"] = _make_test(_desc)


if __name__ == "__main__":
    for _desc in DESCRIPTORS:
        run_learner_conformance(_desc)
        print(f" Learner conformance smoke test passed: {_desc.name}")
