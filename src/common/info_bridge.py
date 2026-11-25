"""进程级信息桥：偶发零散值的 put/get 袋子，不是配置系统。"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

_MISSING = object()


class InfoBridge:
    """进程内自由属性袋。gRPC 工作线程与 Role 主线程共享同一座。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, Any] = {}

    def put(self, **attrs: Any) -> None:
        if not attrs:
            return
        with self._lock:
            self._store.update(attrs)

    def get(self, key: str, default: Any = _MISSING) -> Any:
        with self._lock:
            if default is _MISSING:
                return self._store[key]
            return self._store.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._store)

    @contextmanager
    def scope(self, **overlay: Any) -> Iterator["InfoBridge"]:
        with self._lock:
            saved = dict(self._store)
            if overlay:
                self._store.update(overlay)
        try:
            yield self
        finally:
            with self._lock:
                self._store.clear()
                self._store.update(saved)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


info_bridge = InfoBridge()

INFO_TOML_CONFIG = "toml_config"
INFO_RAW_CONFIG_FILE = "_raw_config_file"
INFO_RAW_CONFIG_NAME = "_raw_config_name"
# 本进程角色身份：True=分布式 master，False=分布式 worker，None=非分布式。
# 仅入口 put，不经 RPC 传播（各进程自行声明）。
INFO_IS_MASTER = "is_master"

_TOML_CONFIG_KEYS = (
    "config_file",
    "design",
    "branch",
    "base_auto_ml_config",
    "train_config",
    "env_config",
    "sim_config",
    "replay_buffer_config",
    "model_manager_config",
    "monte_carlo_method_config",
    "sim_manager_config",
    "rpc_config",
    "agents_config",
    "step_agents_config",
    "learners_config",
    "stage_task_config",
    "game_config",
    "entry_config",
    "eval_config",
    "log_config",
    "plugins_config",
)


def publish_training_info(conf: Any) -> None:
    """Toml 装配完成后把全部配置对象打成一个字典放入桥，键为 toml_config。"""
    info_bridge.put(
        toml_config={key: getattr(conf, key) for key in _TOML_CONFIG_KEYS},
        **{
            INFO_RAW_CONFIG_FILE: conf.raw_file,
            INFO_RAW_CONFIG_NAME: conf.config_file,
        },
    )


def public_training_info(key: str | None = None) -> Any:
    """返回允许经 RPC 传播的信息；原始 TOML 正文、源路径与本进程 IS_MASTER 只留本地。"""
    private = {INFO_RAW_CONFIG_FILE, INFO_RAW_CONFIG_NAME, INFO_IS_MASTER}
    if key in private:
        return None
    if key is not None:
        return info_bridge.get(key, None)
    snapshot = info_bridge.snapshot()
    for name in private:
        snapshot.pop(name, None)
    return snapshot
