"""按需接入可选仿真 SDK；普通训练导入不会加载 dodo_sim。"""
from __future__ import annotations

import importlib
import copy
import sys
import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim import SimManager, SimManagerConfig


def load_sim_module(name: str):
    """仅在启用仿真能力时加载 SDK，保留 SDK 内部缺依赖的原始错误。"""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != "dodo_sim":
            raise
        raise ModuleNotFoundError(
            "仿真功能需要 dodo-sim>=0.4.0,<0.5；请在场景工程的解释器中安装 dodo-sim。",
            name="dodo_sim",
        ) from exc


def create_sim_manager(config: SimManagerConfig | None, *, sample_type: str | None = None) -> SimManager | None:
    if config is None or not config.use_sim or sample_type == "offline":
        return None
    from utils.task_id import task_id
    return load_sim_module("dodo_sim.sim_manager").SimManager(config=config, task_id=str(task_id))


def close_sim_manager_on_init_error(initializer):
    """角色构造失败时立即回收已创建的仿真，不等进程退出。"""
    @functools.wraps(initializer)
    def initialize(self, *args, **kwargs):
        try:
            return initializer(self, *args, **kwargs)
        except BaseException as error:
            manager = getattr(self, "sim_manager", None)
            if manager is not None:
                try:
                    manager.close()
                except Exception as cleanup:
                    error.add_note(f"SimManager cleanup failed: {cleanup}")
            raise
    return initialize


def simulation_exception_types() -> tuple[type[Exception], ...]:
    """SDK 异常产生时其模块已加载；错误处理本身不触发可选包加载。"""
    module = sys.modules.get("dodo_sim.exceptions")
    return (module.SimulationException,) if module is not None else ()


def simulation_parameter_exception_types() -> tuple[type[Exception], ...]:
    module = sys.modules.get("dodo_sim.exceptions")
    return (module.ValueErrorException,) if module is not None else ()


def apply_stage_sim_metadata(manager, task_config):
    """Apply task-owned instance metadata while the stage admission gate is closed."""
    patch = task_config.sim_metadata
    if not isinstance(patch, dict):
        raise TypeError("sim_metadata must be a dictionary")
    if not patch:
        return
    if task_config.sample_type == "offline" or manager is None or manager.instances is None:
        raise ValueError("sim_metadata requires online task-owned simulation instances")
    from importlib.metadata import version
    from packaging.version import Version
    if Version(version("dodo-sim")) < Version("0.4.1"):
        raise RuntimeError("sim_metadata requires dodo-sim>=0.4.1")
    if not hasattr(manager, "_stage_metadata_baseline"):
        manager._stage_metadata_baseline = copy.deepcopy(manager.instances.config.metadata)
    def merge(base, update):
        result = copy.deepcopy(base) if isinstance(base, dict) else {}
        for key, value in update.items():
            result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else copy.deepcopy(value)
        return result
    manager.instances.configure({"metadata": merge(manager._stage_metadata_baseline, patch)})
