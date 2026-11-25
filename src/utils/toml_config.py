from __future__ import annotations

from utils.environment import apply_globe_env_vars, validate_globe_env_vars

import ast
import copy
import operator
import os
import re
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, TypeVar

from config import (
    AgentConfig,
    AutoMLConfig,
    AutoMLOptunaConfig,
    ArtifactsManagerConfig,
    EnvConfig,
    EntryConfig,
    EvalConfig,
    LearnerConfig,
    LocalStorageConfig,
    LogConfig,
    ModelManagerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    GrpcConfig,
    S3StorageConfig,
    StageConfig,
    StageTaskConfig,
    StorageBucket,
    StorageType,
    TrainConfig,
    TorchRpcConfig,
)
from envs.action.model import ActionDefined, StepActionDefined
from scheduler.utils.automl_config import finalize_auto_ml_config
from scheduler.model import (
    AutoMLBusinessMetricConfig,
    AutoMLEnvMetadataSearchParamConfig,
    AutoMLLearnerSearchParamConfig,
    AutoMLMetricOutputConfig,
    AutoMLSeedSearchConfig,
)
from common.exception import PluginNameDuplicateException
from common.logger import configure_logger
from common.plugins import PLUGIN_TYPES, PluginRegistry, PluginsConfig
from common.simulation import load_sim_module
from utils.ini_config import apply_section_to_object, select_device
from utils.task_id import task_id


ConfigT = TypeVar("ConfigT")
_REF_PATTERN = re.compile(r"\$\{([^}]+)}")
_BRANCH_ARM_PATTERN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)(?=\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*=|$)",
)
_TOML_ERROR_LINE_PATTERN = re.compile(r"line\s+(\d+)", re.IGNORECASE)
_TOML_TABLE_PATTERN = re.compile(r"^\s*\[plugins_config\.([A-Za-z_][A-Za-z0-9_]*)\]\s*$")
_TOML_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*=")


def _plugin_duplicate_from_toml_error(
        raw_text: str,
        error: tomllib.TOMLDecodeError,
) -> tuple[str, str] | None:
    """尽可能将 TOML 的重复键错误转换为插件名称重复错误。"""
    line_match = _TOML_ERROR_LINE_PATTERN.search(str(error))
    if line_match is None:
        return None
    line_number = int(line_match.group(1))
    lines = raw_text.splitlines()
    if not 1 <= line_number <= len(lines):
        return None

    plugin_type = ""
    for line in reversed(lines[:line_number]):
        table_match = _TOML_TABLE_PATTERN.match(line.strip())
        if table_match is not None:
            plugin_type = table_match.group(1)
            break
        if line.strip().startswith("["):
            break
    if plugin_type not in PLUGIN_TYPES:
        return None

    key_match = _TOML_KEY_PATTERN.match(lines[line_number - 1])
    if key_match is None:
        return plugin_type, "<table>"
    return plugin_type, key_match.group(1)


@dataclass(frozen=True)
class _TomlSection:
    """一个 TOML 配置段的解析结果；enabled 只在 TOML 层生效。"""

    values: dict[str, Any]
    enabled: bool


class TomlTrainingConfig:
    """
    统一TOML训练配置装配器。

    约定：
    - [design] 是任务设计区，支持 ${name} 引用和 eval: 简单表达式。
    - branch: 按 TomlTrainingConfig(branch=...) 选择一支，右值与 eval: 同一套表达式。
    - agents/learners/stages 使用显式编号表，例如 [agents.0]、[learners.0]、[stages.0]。
    - 训练 Agent 可省略 agent_type，默认值由同名 Learner 的算法配置声明。
    - [agents.N.action_defined] 是每个智能体唯一的四类业务动作定义入口。
    - action_defined.action_key 是编号列表，每项 key 同时作为动作名和动作头名。
    - 旧 [agents.N.actions.M] 继续兼容单头离散 StepActionDefined。
    - [stages.N.learner_overrides.<name>] 基于全局 LearnerConfig 做阶段级覆盖。
    - 可选模块段统一支持 enabled（默认 true）；enabled=false 时不生成对应配置对象。
    - [auto_ml_config] 为 AutoML 全局默认；[stages.N.auto_ml_overrides] 在默认上覆盖。
    """

    def __init__(
            self,
            config_file: str | Path | None = None,
            project_root: Optional[str | Path] = None,
            plugin_registry: Optional[PluginRegistry] = None,
            branch: str = "train",
            *,
            config_text: str | None = None,
    ):
        if (config_file is None) == (config_text is None):
            raise ValueError("Provide exactly one of config_file or config_text")
        self._config_file = Path(config_file) if config_file is not None else None
        self._config_text = config_text
        self._project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        self._plugin_registry = plugin_registry
        self._branch = str(branch or "train").strip() or "train"
        self._build()

    @classmethod
    def read_plugins_config(
            cls,
            config_file: str | Path | None = None,
            project_root: Optional[str | Path] = None,
            *,
            config_text: str | None = None,
    ) -> PluginsConfig:
        """第一阶段只解析插件清单，供 SDK 激活算法插件后再装配强类型配置。"""
        raw_text = cls._read_source(config_file, config_text)
        try:
            raw_config = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as exc:
            duplicate = _plugin_duplicate_from_toml_error(raw_text, exc)
            if duplicate is not None:
                plugin_type, name = duplicate
                raise PluginNameDuplicateException(
                    plugin_type=plugin_type,
                    name=name,
                    existing="same TOML configuration",
                    incoming="same TOML configuration",
                ) from exc
            raise
        probe = cls.__new__(cls)
        probe._project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        probe._branch = "train"
        design = probe._resolve_design(raw_config.get("design", {}))
        resolved = probe._resolve_value({k: v for k, v in raw_config.items() if k != "globe_env_vars"}, design)
        return cls._build_plugins_config(resolved.get("plugins_config", {}))

    @classmethod
    def prepare_runtime_bootstrap(
            cls,
            config_file: str | Path | None = None,
            project_root: Optional[str | Path] = None,
            *,
            config_text: str | None = None,
    ) -> PluginsConfig:
        """轻量解析 design / log / plugins：配置 task_id 与 logger，返回插件清单供激活。"""
        raw_text = cls._read_source(config_file, config_text)
        try:
            raw_config = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as exc:
            duplicate = _plugin_duplicate_from_toml_error(raw_text, exc)
            if duplicate is not None:
                plugin_type, name = duplicate
                raise PluginNameDuplicateException(
                    plugin_type=plugin_type,
                    name=name,
                    existing="same TOML configuration",
                    incoming="same TOML configuration",
                ) from exc
            raise
        apply_globe_env_vars(raw_config.get("globe_env_vars", {}))
        probe = cls.__new__(cls)
        probe._project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path.cwd().resolve()
        )
        probe._branch = "train"
        design = probe._resolve_design(raw_config.get("design", {}))
        task_id.configure(design["task_id"])
        resolved = probe._resolve_value({k: v for k, v in raw_config.items() if k != "globe_env_vars"}, design)
        log_config = probe._build_config(LogConfig, resolved.get("log_config", {}))
        configure_logger(log_config)
        return cls._build_plugins_config(resolved.get("plugins_config", {}))

    @staticmethod
    def _read_source(config_file, config_text):
        if (config_file is None) == (config_text is None):
            raise ValueError("Provide exactly one of config_file or config_text")
        if config_text is not None:
            if not isinstance(config_text, str):
                raise TypeError("config_text must be TOML text")
            return config_text
        return Path(config_file).read_text(encoding="utf-8")

    def _build(self) -> None:
        raw_text = self._read_source(self._config_file, self._config_text)
        self._raw_file = raw_text
        try:
            raw_config = tomllib.loads(raw_text)
        except tomllib.TOMLDecodeError as exc:
            duplicate = _plugin_duplicate_from_toml_error(raw_text, exc)
            if duplicate is not None:
                plugin_type, name = duplicate
                raise PluginNameDuplicateException(
                    plugin_type=plugin_type,
                    name=name,
                    existing="same TOML configuration",
                    incoming="same TOML configuration",
                ) from exc
            raise

        self._design = self._resolve_design(raw_config.get("design", {}))
        task_id.configure(self._design["task_id"])
        self.globe_env_vars = validate_globe_env_vars(raw_config.get("globe_env_vars", {}))
        self._config = self._resolve_value({k: v for k, v in raw_config.items() if k != "globe_env_vars"}, self._design)
        self._config["globe_env_vars"] = dict(self.globe_env_vars)
        self._validate_rpc_sections(raw_config)
        self._plugins_config = self._build_plugins_config(self._config.get("plugins_config", {}))

        if "auto_ml_search_space" in self._config:
            raise ValueError(
                "Top-level [auto_ml_search_space] is removed. "
                "Put numbered entries under [auto_ml_config.learner_search_space.N] "
                "or [stages.N.auto_ml_overrides.learner_search_space.M]."
            )

        base_auto_ml_section = self._read_optional_section("auto_ml_config")
        self._base_auto_ml_values = (
            copy.deepcopy(base_auto_ml_section.values)
            if base_auto_ml_section is not None
            else {}
        )
        self._base_auto_ml_config = (
            _build_auto_ml_config(base_auto_ml_section.values)
            if base_auto_ml_section is not None and base_auto_ml_section.enabled
            else None
        )
        # 构造一次顶层 MC 配置，再深拷贝到各个 TaskConfig；兼容既有
        # [monte_carlo_method_config] TOML，同时保证运行期只有 TaskConfig 字段是配置源。
        self._monte_carlo_method_config = self._build_optional_config(
            "monte_carlo_method_config", MonteCarloMethodConfig,
        )
        self._train_config = self._build_train_config(self._config.get("train_config", {}))
        self._env_config = self._build_config(EnvConfig, self._config.get("env_config", {}))
        self._game_config = self._config.get("game_config", {})
        if self._game_config:
            self._env_config.metadata["game_config"] = copy.deepcopy(self._game_config)

        self._replay_buffer_config = self._build_replay_buffer_config(
            self._config.get("replay_buffer_config", {}),
        )
        self._model_manager_config = self._build_optional_config(
            "model_manager_config", ModelManagerConfig,
        )
        self._artifacts_manager_config = self._build_optional_config(
            "artifacts_manager_config", ArtifactsManagerConfig,
        )
        self._rpc_config = self._build_rpc_config()
        self._entry_config = self._build_entry_config(self._config.get("entry", {}))

        self._base_learners_config = self._build_learners_config()
        self._agents_config = self._build_agents_config()
        self._validate_unpaired_net_agents()
        self._stage_task_config = self._build_stage_task_config()
        self._eval_config = self._build_eval_config(self._config.get("eval_config", {}))
        self._log_config = self._build_config(LogConfig, self._config.get("log_config", {}))
        configure_logger(self._log_config)

    def _load_simulation_config(self, name: str):
        """只装配被访问且启用的配置块，不连带加载其他仿真配置。"""
        section = self._read_optional_section(name)
        if section is not None and not section.enabled:
            return None
        if name == "sim_config":
            if not self._env_config.use_sim:
                return None
        elif section is None or not section.values.get("use_sim", self._env_config.use_sim):
            return None
        # 共享 design / eval / branch 已在统一入口解析，SIM 只负责强类型装配。
        simulation = load_sim_module("dodo_sim.toml_config").TomlSimulationConfig.from_dict(
            {name: section.values if section is not None else {}}, use_sim=self._env_config.use_sim,
        )
        return getattr(simulation, name)

    def _build_entry_config(self, values: dict[str, Any]) -> EntryConfig:
        resolved = self._resolve_value(values, self._design)
        return self._build_config(EntryConfig, resolved if isinstance(resolved, dict) else {})

    def _selected_rpc_protocol(self) -> str | None:
        raw = os.environ.get("DODO_RPC", "").strip().lower()
        if not raw:
            return None
        if raw in ("grpc", "trpc"):
            return raw
        raise ValueError(f"{self.config_file}: DODO_RPC must be grpc or trpc, got {raw!r}")

    def _validate_rpc_sections(self, raw_config: dict[str, Any]) -> None:
        if "rpc_config" in raw_config:
            raise ValueError(
                f"{self.config_file}: [rpc_config] has been removed; "
                "use [grpc_config] or [torch_rpc_config]."
            )
        protocol = self._selected_rpc_protocol()
        has_grpc = "grpc_config" in raw_config
        has_torch = "torch_rpc_config" in raw_config
        if protocol == "grpc":
            if not has_grpc:
                raise ValueError(f"{self.config_file}: DODO_RPC=grpc requires [grpc_config]")
            return
        if protocol == "trpc":
            if not has_torch:
                raise ValueError(f"{self.config_file}: DODO_RPC=trpc requires [torch_rpc_config]")
            return
        if has_grpc and has_torch:
            # 未设置 DODO_RPC 时沿用 torch（科目默认）；显式 grpc 启动脚本会 export DODO_RPC=grpc。
            return

    def _build_rpc_config(self) -> GrpcConfig | TorchRpcConfig:
        protocol = self._selected_rpc_protocol()
        if protocol == "grpc" or "torch_rpc_config" not in self._config:
            return self._build_config(GrpcConfig, self._config.get("grpc_config", {}))
        config = self._build_config(TorchRpcConfig, self._config["torch_rpc_config"])
        if int(config.world_size) < 2:
            raise ValueError("[torch_rpc_config].world_size must be at least 2")
        if not isinstance(config.init_address, str) or not config.init_address.strip():
            raise ValueError("[torch_rpc_config].init_address must be a non-empty host:port")
        address = config.init_address.removeprefix("tcp://")
        host, separator, port = address.rpartition(":")
        if not separator or not host or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError("[torch_rpc_config].init_address must use host:port format")
        config.init_address = address
        if int(config.num_worker_threads) <= 0:
            raise ValueError("[torch_rpc_config].num_worker_threads must be positive")
        if (
                isinstance(config.max_concurrent_replay_pushes, bool)
                or not isinstance(config.max_concurrent_replay_pushes, int)
                or config.max_concurrent_replay_pushes <= 0
        ):
            raise ValueError("[torch_rpc_config].max_concurrent_replay_pushes must be a positive integer")
        config.max_concurrent_replay_pushes = int(config.max_concurrent_replay_pushes)
        if (
                isinstance(config.replay_push_permit_poll_interval_seconds, bool)
                or not 0.01 <= float(config.replay_push_permit_poll_interval_seconds) <= 60.0
        ):
            raise ValueError(
                "[torch_rpc_config].replay_push_permit_poll_interval_seconds must be in [0.01, 60]"
            )
        config.replay_push_permit_poll_interval_seconds = float(
            config.replay_push_permit_poll_interval_seconds
        )
        if (
                isinstance(config.replay_push_permit_lease_timeout_seconds, bool)
                or not 1.0 <= float(config.replay_push_permit_lease_timeout_seconds) <= 86400.0
        ):
            raise ValueError(
                "[torch_rpc_config].replay_push_permit_lease_timeout_seconds must be in [1, 86400]"
            )
        config.replay_push_permit_lease_timeout_seconds = float(
            config.replay_push_permit_lease_timeout_seconds
        )
        chunk_bytes = config.replay_push_chunk_bytes
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int):
            raise ValueError("[torch_rpc_config].replay_push_chunk_bytes must be an integer")
        if chunk_bytes != 0 and not 1048576 <= chunk_bytes <= 536870912:
            raise ValueError(
                "[torch_rpc_config].replay_push_chunk_bytes must be 0 or in [1048576, 536870912]"
            )
        config.replay_push_chunk_bytes = int(chunk_bytes)
        return config

    @staticmethod
    def _build_plugins_config(values: Any) -> PluginsConfig:
        if not values:
            return PluginsConfig()
        if not isinstance(values, dict):
            raise TypeError("[plugins_config] must be a TOML table")
        enabled = values.get("enabled", [])
        disabled = values.get("disabled", [])
        if not isinstance(enabled, list) or not all(isinstance(item, str) and item for item in enabled):
            raise TypeError("[plugins_config].enabled must be a list of non-empty plugin ids")
        if not isinstance(disabled, list) or not all(isinstance(item, str) and item for item in disabled):
            raise TypeError("[plugins_config].disabled must be a list of non-empty plugin ids")
        if enabled and disabled:
            raise ValueError("[plugins_config].enabled and .disabled are mutually exclusive")
        unknown_types = set(values) - {"enabled", "disabled", *PLUGIN_TYPES}
        if unknown_types:
            raise ValueError(f"[plugins_config] contains unsupported plugin types: {sorted(unknown_types)}")
        locations: dict[str, dict[str, str]] = {plugin_type: {} for plugin_type in PLUGIN_TYPES}
        for plugin_type in PLUGIN_TYPES:
            raw_locations = values.get(plugin_type, {})
            if not isinstance(raw_locations, dict):
                raise TypeError(f"[plugins_config.{plugin_type}] must be a TOML table")
            for name, location in raw_locations.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"[plugins_config.{plugin_type}] contains an empty plugin name")
                if not isinstance(location, str) or ":" not in location:
                    raise ValueError(
                        f"Plugin '{plugin_type}.{name}' must be a class location: module:ClassName"
                    )
                locations[plugin_type][name] = location
        return PluginsConfig(enabled=list(enabled), disabled=list(disabled), locations=locations)

    def _build_eval_config(self, values: dict[str, Any]) -> EvalConfig:
        resolved = self._resolve_value(values, self._design)
        if not isinstance(resolved, dict):
            resolved = {}
        metric_outputs_raw = resolved.pop("metric_outputs", {})
        monte_carlo_values = resolved.pop("monte_carlo_method_config", None)
        eval_config = self._build_config(EvalConfig, resolved)
        self._apply_monte_carlo_config(eval_config, monte_carlo_values)
        eval_config.metric_outputs = {
            name: output.to_dict()
            for name, output in _build_metric_outputs(metric_outputs_raw or {}).items()
        }
        return eval_config

    def _resolve_design(self, raw_design: dict[str, Any]) -> dict[str, Any]:
        design = copy.deepcopy(raw_design)
        design.setdefault("task_id", "")
        if design.get("root_dir") in ("{{root_dir}}", "{{project_root}}"):
            design["root_dir"] = str(self._project_root)

        resolved: dict[str, Any] = {}
        pending = dict(design)
        while pending:
            progressed = False
            for key in list(pending.keys()):
                try:
                    resolved[key] = self._resolve_value(pending[key], resolved)
                except KeyError:
                    continue
                pending.pop(key)
                progressed = True
            if not progressed:
                unresolved = ", ".join(sorted(pending))
                raise ValueError(f"TOML design variables cannot be resolved: {unresolved}")
        return resolved

    def _resolve_value(self, value: Any, design: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._resolve_value(item, design) for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, design) for item in value]
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if raw.startswith("branch:"):
            return self._resolve_branch(raw[len("branch:"):].strip(), design)
        if raw.startswith("eval:"):
            return _safe_eval(raw[len("eval:"):].strip(), design)

        exact_match = _REF_PATTERN.fullmatch(raw)
        if exact_match:
            key = exact_match.group(1).strip()
            if key not in design:
                raise KeyError(key)
            return design[key]

        def replace(match: re.Match[str]) -> str:
            key = match.group(1).strip()
            if key not in design:
                raise KeyError(key)
            return str(design[key])

        return _REF_PATTERN.sub(replace, value)

    def _resolve_branch(self, spec: str, design: dict[str, Any]) -> Any:
        arms = {
            match.group(1): match.group(2).strip()
            for match in _BRANCH_ARM_PATTERN.finditer(spec)
        }
        if not arms:
            raise ValueError(f"Invalid branch: expression: {spec!r}")
        branch = getattr(self, "_branch", "train")
        if branch not in arms:
            raise ValueError(
                f"branch={branch!r} is not defined in branch: expression; "
                f"available={sorted(arms)}"
            )
        return self._resolve_branch_arm(arms[branch], design)

    def _resolve_branch_arm(self, arm: str, design: dict[str, Any]) -> Any:
        arm = arm.strip()
        if arm.startswith("eval:") or arm.startswith("branch:") or _REF_PATTERN.search(arm):
            return self._resolve_value(arm, design)
        try:
            return _safe_eval(arm, design)
        except (SyntaxError, ValueError):
            return arm

    def _build_train_config(self, values: dict[str, Any]) -> TrainConfig:
        values = _normalize_fields(copy.deepcopy(values))
        auto_ml_overrides = values.pop("auto_ml_overrides", None)
        if auto_ml_overrides is None:
            auto_ml_overrides = values.pop("auto_ml_config", None)
        monte_carlo_values = values.pop("monte_carlo_method_config", None)
        config = self._build_config(TrainConfig, values)
        self._apply_monte_carlo_config(config, monte_carlo_values)
        if auto_ml_overrides is not None:
            section = _parse_optional_section(
                auto_ml_overrides,
                "[train_config.auto_ml_overrides]",
            )
            if section.enabled:
                config.auto_ml_config = _merge_auto_ml_overrides(
                    self._base_auto_ml_values,
                    section.values,
                )
        elif self._base_auto_ml_config is not None:
            config.auto_ml_config = copy.deepcopy(self._base_auto_ml_config)
        return config

    def _apply_monte_carlo_config(self, task_config, values) -> None:
        """把顶层 MC 深拷贝到 TaskConfig，并翻译可选的嵌套阶段覆盖。"""
        if self._monte_carlo_method_config is not None:
            task_config.monte_carlo_method_config = copy.deepcopy(self._monte_carlo_method_config)
        if values is None:
            return
        section = _parse_optional_section(values, "[monte_carlo_method_config]")
        if not section.enabled:
            task_config.monte_carlo_method_config = None
            return
        if task_config.monte_carlo_method_config is None:
            task_config.monte_carlo_method_config = MonteCarloMethodConfig()
        apply_section_to_object(
            task_config.monte_carlo_method_config,
            _normalize_fields(copy.deepcopy(section.values)),
        )
        task_config.monte_carlo_method_config.__post_init__()

    def _read_optional_section(self, key: str) -> Optional[_TomlSection]:
        if key not in self._config:
            return None
        return _parse_optional_section(self._config[key], f"[{key}]")

    def _build_optional_config(
            self,
            key: str,
            config_cls: type[ConfigT],
    ) -> Optional[ConfigT]:
        """可选 TOML 段的统一构造入口；缺段或 enabled=false 都返回 None。"""
        section = self._read_optional_section(key)
        if section is None or not section.enabled:
            return None
        return self._build_config(config_cls, section.values)

    def _build_config(self, config_cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
        values = _normalize_fields(copy.deepcopy(values))
        config = config_cls()
        apply_section_to_object(config, values)
        return config

    def _build_replay_buffer_config(self, values: dict[str, Any]) -> ReplayBufferConfig:
        """嵌套 [replay_buffer_config.storage] 是 dict，落到 dataclass 的 storage 对象上。"""
        values = _normalize_fields(copy.deepcopy(values))
        storage_values = values.pop("storage", None)
        config = ReplayBufferConfig()
        apply_section_to_object(config, values)
        config.__post_init__()
        if isinstance(storage_values, dict):
            apply_section_to_object(config.storage, _normalize_fields(storage_values))
            config.__post_init__()
        return config

    def _build_agents_config(self) -> list[AgentConfig]:
        agents_config: list[AgentConfig] = []
        learners_by_name = {learner.name: learner for learner in self._base_learners_config}
        for _, values in self._indexed_items("agents"):
            values = _normalize_fields(copy.deepcopy(values))
            actions = values.pop("actions", {})
            action_defined_values = values.pop("action_defined", None)
            algo_config_values = values.pop("algo_config", {})
            name = values.pop("name")
            agent_type = values.pop("agent_type", None)
            if agent_type is None:
                learner = learners_by_name.get(name)
                if learner is None:
                    raise ValueError(f"Agent {name!r} has no matching Learner. Set agent_type explicitly.")
                agent_type = type(learner.algo_config).default_agent_type
                if not agent_type:
                    raise ValueError(
                        f"Learner {learner.algo_type!r} has no default Agent type. "
                        f"Set agent_type explicitly for Agent {name!r}."
                    )
            algorithm_spec = self._algorithm_spec("agent", agent_type)
            algo_config = algorithm_spec.config_class()
            if algo_config_values:
                apply_section_to_object(algo_config, _normalize_fields(algo_config_values))
            agent_config = AgentConfig(name=name, agent_type=agent_type, algo_config=algo_config)
            if algorithm_spec.default_obs_agent_id is not None:
                agent_config.obs_agent_id = algorithm_spec.default_obs_agent_id
            apply_section_to_object(agent_config, values)
            agent_config.define_action = [
                StepActionDefined.from_dict(_normalize_action_values(action_values))
                for _, action_values in _indexed_items_from(actions)
            ]
            if isinstance(action_defined_values, dict):
                agent_config.action_defined = ActionDefined.from_dict(
                    _normalize_action_definition_values(action_defined_values)
                )
            agents_config.append(agent_config)
        return agents_config

    def _build_learners_config(self) -> list[LearnerConfig]:
        learners_config: list[LearnerConfig] = []
        for _, values in self._indexed_items("learners"):
            values = _normalize_fields(copy.deepcopy(values))
            algo_config_values = values.pop("algo_config", {})
            constructor_values = {
                "name": values.pop("name"),
                "algo_type": values.pop("algo_type"),
            }
            algorithm_spec = self._algorithm_spec("learner", constructor_values["algo_type"])
            algo_config = algorithm_spec.config_class()
            apply_section_to_object(algo_config, _normalize_fields(algo_config_values))
            learner_config = LearnerConfig(**constructor_values, algo_config=algo_config)
            apply_section_to_object(learner_config, values)
            learners_config.append(learner_config)
        return learners_config

    def _algorithm_spec(self, kind: str, name: str):
        if self._plugin_registry is not None:
            return self._plugin_registry.algorithm_spec(kind, name)

        # Portal 和旧的直接解析调用仍可使用内置配置；自定义算法由 SDK 两阶段加载提供。
        from common.plugins import AlgorithmPluginSpec
        from config import AGENT_ALGO_CONFIG, AGENT_DEFAULT_OBS_AGENT_ID, LEARNER_ALGO_CONFIG
        if kind == "agent":
            try:
                agent_type = next(item for item in AGENT_ALGO_CONFIG if item.value == name)
            except StopIteration:
                raise ValueError(
                    f"Custom agent plugin {name!r} requires an activated PluginRegistry"
                ) from None
            return AlgorithmPluginSpec(
                AGENT_ALGO_CONFIG[agent_type],
                AGENT_DEFAULT_OBS_AGENT_ID[agent_type],
            )
        try:
            algo_type = next(item for item in LEARNER_ALGO_CONFIG if item.value == name)
        except StopIteration:
            raise ValueError(
                f"Custom learner plugin {name!r} requires an activated PluginRegistry"
            ) from None
        return AlgorithmPluginSpec(LEARNER_ALGO_CONFIG[algo_type])

    def _validate_unpaired_net_agents(self) -> None:
        """网络型 Agent 若没有同名 Learner，必须自带 model_file_path；不改写任何配置字段。"""
        learners_by_name = {learner.name: learner for learner in self._base_learners_config}
        for agent_config in self._agents_config:
            if agent_config.name in learners_by_name:
                continue
            if agent_config.algo_config.use_net:
                assert str(agent_config.model_file_path or "").strip(), (
                    f"未配置 Learner 的网络型 AgentConfig 必须提供 model_file_path，"
                    f"Agent={agent_config.name}，Learner={sorted(learners_by_name)}"
                )

    def _build_stage_task_config(self) -> StageTaskConfig:
        stage_task_values = copy.deepcopy(self._config.get("stage_task_config", {}))
        stages: list[StageConfig] = []
        base_learners_by_name = {learner.name: learner for learner in self._base_learners_config}
        for _, stage_values in self._indexed_items("stages"):
            values = _normalize_fields(copy.deepcopy(stage_values))
            stage_name = str(values.pop("name"))
            learner_names = values.pop("learners", list(base_learners_by_name))
            train_overrides = values.pop("train_overrides", {})
            agent_overrides = values.pop("agent_overrides", {})
            learner_overrides = values.pop("learner_overrides", {})
            clear_replay_buffer = values.pop("clear_replay_buffer", True)
            rehearsal_archive_ratio = float(values.pop("rehearsal_archive_ratio", 0.0))
            rehearsal_archive_select = str(
                values.pop("rehearsal_archive_select", "even_span")
            ).strip() or "even_span"
            rehearsal_sample_ratio = float(values.pop("rehearsal_sample_ratio", 0.0))
            save_replay_buffer = bool(values.pop("save_replay_buffer", False))
            auto_ml_overrides = values.pop("auto_ml_overrides", None)
            if auto_ml_overrides is None:
                auto_ml_overrides = values.pop("auto_ml_config", None)
            assert not values, (
                f"Stage '{stage_name}' contains unsupported fields {sorted(values)}. "
                f"TrainConfig overrides must be written under [stages.N.train_overrides], "
                f"Agent overrides under [stages.N.agent_overrides.<agent_name>], "
                f"Learner overrides under [stages.N.learner_overrides.<learner_name>], "
                f"AutoML overrides under [stages.N.auto_ml_overrides]."
            )
            assert 0.0 <= rehearsal_archive_ratio <= 1.0, (
                f"Stage '{stage_name}' rehearsal_archive_ratio must be in [0, 1], "
                f"got {rehearsal_archive_ratio}"
            )
            assert 0.0 <= rehearsal_sample_ratio <= 1.0, (
                f"Stage '{stage_name}' rehearsal_sample_ratio must be in [0, 1], "
                f"got {rehearsal_sample_ratio}"
            )
            assert rehearsal_archive_select in {"even_span", "newest", "uniform"}, (
                f"Stage '{stage_name}' rehearsal_archive_select must be one of "
                f"even_span|newest|uniform, got {rehearsal_archive_select!r}"
            )

            stage_train_config = copy.deepcopy(self._train_config)
            train_overrides = _normalize_fields(copy.deepcopy(train_overrides))
            stage_mc_values = train_overrides.pop("monte_carlo_method_config", None)
            apply_section_to_object(stage_train_config, train_overrides)
            self._apply_monte_carlo_config(stage_train_config, stage_mc_values)
            stage_auto_ml_config = None
            if auto_ml_overrides is not None:
                section = _parse_optional_section(
                    auto_ml_overrides,
                    f"[stages.{stage_name}.auto_ml_overrides]",
                )
                if section.enabled:
                    stage_auto_ml_config = _merge_auto_ml_overrides(
                        self._base_auto_ml_values,
                        section.values,
                    )
            stage_agents = self._build_stage_agents(stage_name, agent_overrides)

            stage_learners: list[LearnerConfig] = []
            for learner_name in learner_names:
                assert learner_name in base_learners_by_name, (
                    f"Stage {stage_name} references unknown learner '{learner_name}', "
                    f"available={sorted(base_learners_by_name)}"
                )
                learner_config = copy.deepcopy(base_learners_by_name[learner_name])
                override_values = copy.deepcopy(learner_overrides.get(learner_name, {}))
                algo_override = override_values.pop("algo_config", {})
                apply_section_to_object(learner_config, _normalize_fields(override_values))
                apply_section_to_object(learner_config.algo_config, _normalize_fields(algo_override))
                stage_learners.append(learner_config)

            stages.append(StageConfig(
                stage_name=stage_name,
                train_config=stage_train_config,
                agents_config=stage_agents,
                learners_config=stage_learners,
                clear_replay_buffer=clear_replay_buffer,
                rehearsal_archive_ratio=rehearsal_archive_ratio,
                rehearsal_archive_select=rehearsal_archive_select,
                rehearsal_sample_ratio=rehearsal_sample_ratio,
                save_replay_buffer=save_replay_buffer,
                auto_ml_config=stage_auto_ml_config,
            ))
        skip_stage_model_load_dir = stage_task_values.get("skip_stage_model_load_dir")
        skip_stage_model_version = stage_task_values.get("skip_stage_model_version")
        skip_stage_replay_path = stage_task_values.get("skip_stage_replay_path")
        stage_model_versions = copy.deepcopy(stage_task_values.get("stage_model_versions", []))
        # 兼容旧 TOML：仅有 stage_model_versions 且未写 skip_stage_model_version 时映射 [0]
        if (
                skip_stage_model_version is None
                and stage_model_versions
                and int(stage_task_values.get("start_stage", 0) or 0) > 0
        ):
            skip_stage_model_version = stage_model_versions[0]
        return StageTaskConfig(
            stages=stages,
            bootstrap_learners_config=copy.deepcopy(self._base_learners_config),
            start_stage=stage_task_values.get("start_stage", 0),
            stage_model_versions=stage_model_versions,
            skip_stage_model_load_dir=skip_stage_model_load_dir,
            skip_stage_model_version=skip_stage_model_version,
            skip_stage_replay_path=skip_stage_replay_path,
        )

    def _build_stage_agents(self, stage_name: str, agent_overrides: dict[str, Any]) -> list[AgentConfig]:
        """从全局AgentConfig派生本阶段AgentConfig，并应用 [stages.N.agent_overrides.<name>]。"""
        stage_agents = copy.deepcopy(self._agents_config)
        stage_agents_by_name = {agent.name: agent for agent in stage_agents}
        for agent_name, override_values in agent_overrides.items():
            assert agent_name in stage_agents_by_name, (
                f"Stage {stage_name} references unknown agent '{agent_name}', "
                f"available={sorted(stage_agents_by_name)}"
            )
            override_values = _normalize_fields(copy.deepcopy(override_values))
            algo_override = override_values.pop("algo_config", {})
            apply_section_to_object(stage_agents_by_name[agent_name], override_values)
            apply_section_to_object(stage_agents_by_name[agent_name].algo_config, algo_override)
        return stage_agents

    def _indexed_items(self, key: str) -> list[tuple[str, dict[str, Any]]]:
        return _indexed_items_from(self._config.get(key, {}))

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        """保留兼容入口；运行期根目录统一由当前 working directory 决定。"""
        del start
        return Path.cwd().resolve()

    @property
    def config_file(self) -> str:
        return str(self._config_file.resolve()) if self._config_file is not None else "training.toml"

    @property
    def raw_file(self) -> str:
        return self._raw_file

    @property
    def design(self) -> dict[str, Any]:
        return self._design

    @property
    def branch(self) -> str:
        return self._branch

    @property
    def base_auto_ml_config(self) -> Optional[AutoMLConfig]:
        return self._base_auto_ml_config

    @property
    def train_config(self) -> TrainConfig:
        return self._train_config

    @property
    def env_config(self) -> EnvConfig:
        return self._env_config

    @cached_property
    def sim_config(self):
        """按需取得客户端仿真配置；纯 game 场景不加载 dodo-sim。"""
        return self._load_simulation_config("sim_config")

    @cached_property
    def sim_manager_config(self):
        """按需取得中心资源池配置；可显式注入对象覆盖 TOML 装配。"""
        return self._load_simulation_config("sim_manager_config")

    @property
    def replay_buffer_config(self) -> ReplayBufferConfig:
        return self._replay_buffer_config

    @property
    def model_manager_config(self) -> Optional[ModelManagerConfig]:
        return self._model_manager_config

    @property
    def monte_carlo_method_config(self) -> Optional[MonteCarloMethodConfig]:
        return self._monte_carlo_method_config

    @property
    def artifacts_manager_config(self) -> Optional[ArtifactsManagerConfig]:
        """可选；缺省 None，由 build_artifacts_manager 回退到 TaskConfig.artifacts_archive_save_dir。"""
        return self._artifacts_manager_config

    @property
    def rpc_config(self) -> GrpcConfig | TorchRpcConfig:
        return self._rpc_config

    @property
    def agents_config(self) -> list[AgentConfig]:
        return self._agents_config

    @property
    def step_agents_config(self) -> list[AgentConfig]:
        return self._agents_config

    @property
    def learners_config(self) -> list[LearnerConfig]:
        return self._base_learners_config

    @property
    def stage_task_config(self) -> StageTaskConfig:
        return self._stage_task_config

    @property
    def game_config(self) -> dict[str, Any]:
        return self._game_config

    @property
    def entry_config(self) -> EntryConfig:
        return self._entry_config

    @property
    def eval_config(self) -> EvalConfig:
        return self._eval_config

    @property
    def log_config(self) -> LogConfig:
        return self._log_config

    @property
    def plugins_config(self) -> PluginsConfig:
        return self._plugins_config


def _build_indexed_search_space(values: Any, config_cls, section_name: str) -> list:
    if not values:
        return []
    if not isinstance(values, dict):
        raise TypeError(f"{section_name} must be an indexed TOML table")
    indexed = _indexed_items_from(values)
    unknown = sorted(str(key) for key in values if not str(key).isdigit())
    if unknown:
        raise ValueError(
            f"{section_name} only supports numbered entries such as '.0'; "
            f"unsupported keys={unknown}"
        )
    result = []
    for key, raw in indexed:
        if not isinstance(raw, dict):
            raise TypeError(f"{section_name}.{key} must be a TOML table")
        item = config_cls()
        apply_section_to_object(item, _normalize_fields(copy.deepcopy(raw)))
        result.append(item)
    return result


def _flatten_legacy_auto_ml_sections(values: dict[str, Any]) -> dict[str, Any]:
    """把旧版 monitor/metric/archive 嵌套表合并为扁平 AutoMLConfig 字段。"""
    flattened = dict(values)
    monitor_values = flattened.pop("monitor", {})
    if isinstance(monitor_values, dict):
        flattened.update(monitor_values)

    metric_values = flattened.pop("metric", {})
    if isinstance(metric_values, dict):
        metric_values = dict(metric_values)
        if "primary" in metric_values:
            flattened["metric_primary"] = metric_values.pop("primary")
        flattened.update(metric_values)

    archive_values = flattened.pop("archive", {})
    if isinstance(archive_values, dict):
        archive_values = dict(archive_values)
        storage_values = archive_values.pop("storage", None)
        for key, value in archive_values.items():
            target_key = key if key.startswith("archive_") else f"archive_{key}"
            flattened[target_key] = value
        if isinstance(storage_values, dict):
            flattened["storage"] = storage_values

    if "archive_storage_type" in flattened and "storage_type" not in flattened:
        flattened["storage_type"] = flattened.pop("archive_storage_type")

    storage_overrides = flattened.pop("storage", None)
    if not isinstance(storage_overrides, dict):
        storage_overrides = {}
    legacy_storage_map = {
        "archive_endpoint_url": "endpoint_url",
        "archive_access_key": "access_key",
        "archive_secret_key": "secret_key",
        "archive_bucket_name": "bucket_name",
    }
    for legacy_key, storage_key in legacy_storage_map.items():
        if legacy_key in flattened:
            storage_overrides[storage_key] = flattened.pop(legacy_key)
    if storage_overrides:
        flattened["storage"] = storage_overrides

    for legacy_key in (
            "archive_include_artifacts",
            "archive_force_checkpoint_before_archive",
            "archive_write_metrics_csv",
            "archive_write_advice_yaml",
    ):
        flattened.pop(legacy_key, None)

    optuna_values = flattened.get("optuna")
    if isinstance(optuna_values, dict):
        optuna_values = dict(optuna_values)
        sampler_values = optuna_values.pop("sampler", None)
        pruner_values = optuna_values.pop("pruner", None)
        if isinstance(sampler_values, dict):
            if "name" in sampler_values:
                optuna_values["sampler_name"] = sampler_values["name"]
            if "n_startup_trials" in sampler_values:
                optuna_values["sampler_n_startup_trials"] = sampler_values["n_startup_trials"]
        if isinstance(pruner_values, dict):
            rename = {
                "name": "pruner_name",
                "n_startup_trials": "pruner_n_startup_trials",
                "n_warmup_steps": "pruner_n_warmup_steps",
                "min_resource": "pruner_min_resource",
                "max_resource": "pruner_max_resource",
                "reduction_factor": "pruner_reduction_factor",
            }
            for old_key, new_key in rename.items():
                if old_key in pruner_values:
                    optuna_values[new_key] = pruner_values[old_key]
        flattened["optuna"] = optuna_values
    return flattened


def _apply_auto_ml_storage_overrides(config: AutoMLConfig, storage_values: dict[str, Any]) -> None:
    """按 storage_type 写入 storage；local 模式忽略 S3 专用字段（TOML 可预置 endpoint 待切换）。"""
    if not storage_values:
        return
    normalized = _normalize_fields(storage_values)
    storage_type = StorageType(config.storage_type)
    if storage_type == StorageType.S3:
        if not isinstance(config.storage, S3StorageConfig):
            config.storage = S3StorageConfig(bucket_name=StorageBucket.AUTOML.value)
        apply_section_to_object(config.storage, normalized)
        return
    if isinstance(config.storage, LocalStorageConfig) and "data_path" in normalized:
        apply_section_to_object(config.storage, {"data_path": normalized["data_path"]})


def _build_auto_ml_config(values: dict[str, Any]) -> AutoMLConfig:
    values = _normalize_fields(copy.deepcopy(values))
    values = _flatten_legacy_auto_ml_sections(values)
    search_space_raw = values.pop("learner_search_space", {})
    env_search_space_raw = values.pop("env_metadata_search_space", {})
    seed_search_raw = values.pop("seed_search", None)
    metric_outputs_raw = values.pop("metric_outputs", {})
    business_metric_raw = values.pop("business_metric", None)
    optuna_values = values.pop("optuna", {})
    storage_values = values.pop("storage", None)

    optuna = AutoMLOptunaConfig()
    apply_section_to_object(optuna, optuna_values)

    config = AutoMLConfig(**values)
    apply_section_to_object(config, values)
    if isinstance(storage_values, dict):
        _apply_auto_ml_storage_overrides(config, storage_values)
    config.optuna = optuna
    config.learner_search_space = _build_indexed_search_space(
        search_space_raw,
        AutoMLLearnerSearchParamConfig,
        "learner_search_space",
    )
    config.env_metadata_search_space = _build_indexed_search_space(
        env_search_space_raw,
        AutoMLEnvMetadataSearchParamConfig,
        "env_metadata_search_space",
    )
    if seed_search_raw is not None:
        section = _parse_optional_section(seed_search_raw, "seed_search")
        if section.enabled:
            config.seed_search = AutoMLSeedSearchConfig()
            apply_section_to_object(config.seed_search, section.values)
    config.metric_outputs = _build_metric_outputs(metric_outputs_raw)
    if business_metric_raw is not None:
        section = _parse_optional_section(business_metric_raw, "business_metric")
        if section.enabled:
            config.business_metric = AutoMLBusinessMetricConfig(enabled=True)
            apply_section_to_object(config.business_metric, section.values)
    finalize_auto_ml_config(config)
    return config


def _build_metric_outputs(values: Any) -> dict[str, AutoMLMetricOutputConfig]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise TypeError("metric_outputs must be a TOML table")
    outputs: dict[str, AutoMLMetricOutputConfig] = {}
    for name, raw in values.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"metric_outputs name must be a valid identifier: {name!r}")
        if not isinstance(raw, dict):
            raise TypeError(f"metric_outputs.{name} must be a TOML table")
        item = AutoMLMetricOutputConfig()
        apply_section_to_object(item, _normalize_fields(copy.deepcopy(raw)))
        outputs[name] = item
    return outputs


def _parse_optional_section(values: Any, section_name: str) -> _TomlSection:
    """提取 TOML 专用 enabled，并保留其余字段供对应 Config 构造。"""
    if not isinstance(values, dict):
        raise TypeError(f"{section_name} must be a TOML table, given={type(values).__name__}")
    normalized = _normalize_fields(copy.deepcopy(values))
    enabled = normalized.pop("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError(f"{section_name}.enabled must be bool, given={type(enabled).__name__}")
    return _TomlSection(values=normalized, enabled=enabled)


def _merge_auto_ml_overrides(
        base_values: dict[str, Any],
        overrides: dict[str, Any],
) -> AutoMLConfig:
    """在原始 TOML 值层合并 AutoML 默认项与阶段覆盖，再构造唯一的运行时配置。"""
    merged = _merge_mapping(base_values, overrides)
    # 编号表在阶段层表达完整列表；不能递归合并后把全局默认的多余编号遗留进来。
    for key in ("learner_search_space", "env_metadata_search_space", "metric_outputs"):
        if key in overrides:
            merged[key] = copy.deepcopy(overrides[key])
    return _build_auto_ml_config(merged)


def _merge_mapping(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """递归合并 TOML 表；覆盖值只影响自身分支。"""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _indexed_items_from(values: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = [(key, value) for key, value in values.items() if str(key).isdigit()]
    return sorted(items, key=lambda item: int(item[0]))


def _normalize_action_values(action_values: dict[str, Any]) -> dict[str, Any]:
    """将 [agents.N.actions.M.parameters.K] 编号表归一为 StepActionDefined 所需的列表。"""
    values = _normalize_fields(copy.deepcopy(action_values))
    parameters = values.get("parameters")
    if isinstance(parameters, dict):
        values["parameters"] = [
            param_values for _, param_values in _indexed_items_from(parameters)
        ]
    return values


def _normalize_action_definition_values(definition_values: dict[str, Any]) -> dict[str, Any]:
    """保留 action_key 字典顺序；其内部编号参数由 ActionDefined 解析。"""
    return _normalize_fields(copy.deepcopy(definition_values))


def _normalize_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_fields_by_key(key, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_fields(item) for item in value]
    return value


def _normalize_seed_value(value: Any) -> Optional[int]:
    """与 INI 约定一致：'none'/空串表示不固定随机性。"""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in ["none", "None", "null", "Null"]:
            return None
        value = stripped
    seed_value = int(value) % (2 ** 32 - 1)
    return seed_value if seed_value > 0 else 1


def _normalize_fields_by_key(key: str, value: Any) -> Any:
    normalized = _normalize_fields(value)
    if key == "device":
        return select_device(normalized)
    if key == "seed":
        return _normalize_seed_value(normalized)
    if key == "model_version" and isinstance(normalized, str):
        stripped = normalized.strip()
        return None if stripped.lower() in {"", "none", "null"} else stripped
    return normalized


def _safe_eval(expression: str, variables: dict[str, Any]) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body, variables)


def _eval_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        value = variables[node.id]
        if isinstance(value, str) and (value.startswith("eval:") or _REF_PATTERN.search(value)):
            raise KeyError(node.id)
        return value
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        op_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }
        op_type = type(node.op)
        if op_type in op_map:
            return op_map[op_type](left, right)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name) and node.func.id == "len", (
            f"Only len(...) is allowed in TOML eval expression, got={ast.dump(node)}"
        )
        assert len(node.args) == 1 and not node.keywords, "len(...) accepts exactly one positional argument."
        return len(_eval_node(node.args[0], variables))
    raise ValueError(f"Unsupported TOML eval expression node: {ast.dump(node)}")
