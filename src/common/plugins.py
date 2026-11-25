"""TOML 驱动的运行期插件注册与资源解析。"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from common.logger import init_logger
from common.exception import PluginImplementationDuplicateException, PluginNameDuplicateException
from common.registry import (
    clear_project_overrides,
    configure_project_overrides,
    find_extensible_base,
)

LOG = init_logger("plugins")

RESOURCE_KINDS = ("env", "sim", "agent", "learner")
RETIRED_LEARNERS = frozenset({"ia2c", "ippo", "iql", "maa2c", "pac", "pac_dcg", "vdn"})
PLUGIN_TYPES = {
    "simulation": "sim",
    "environment": "env",
    "agent": "agent",
    "learner": "learner",
    "module": None,
}
RESOURCE_PLUGIN_TYPES = {
    resource_kind: plugin_type
    for plugin_type, resource_kind in PLUGIN_TYPES.items()
    if resource_kind is not None
}
_BUNDLED_RUNTIME_ROOTS = {
    "algos", "common", "config", "engine", "envs", "models",
    "roles", "rpc", "sample", "scheduler", "utils",
}


def _canonical_plugin_location(location: str) -> str:
    """比较新旧内置插件路径时忽略安装版运行时命名空间。"""
    prefix = "dodorl_sdk.dodorl."
    return location.removeprefix(prefix)


@dataclass(frozen=True)
class AlgorithmPluginSpec:
    """Agent/Learner 实现与其强类型算法配置的联合声明。"""

    config_class: type
    default_obs_agent_id: bool | None = None


def _algorithm_plugin(
        kind: str,
        config_class: type,
        *,
        default_obs_agent_id: bool | None = None,
) -> Callable[[type], type]:
    if kind not in {"agent", "learner"}:
        raise ValueError(f"Unsupported algorithm plugin kind: {kind}")

    def decorate(implementation: type) -> type:
        if not isinstance(implementation, type):
            raise TypeError(f"{kind}_plugin can only decorate a class")
        implementation.__dodo_algorithm_plugin__ = AlgorithmPluginSpec(
            config_class=config_class,
            default_obs_agent_id=default_obs_agent_id,
        )
        return implementation

    return decorate


def agent_plugin(
        config_class: type,
        *,
        default_obs_agent_id: bool = True,
) -> Callable[[type], type]:
    """联合声明 Agent 实现、AgentAlgoConfig 类型及 obs_agent_id 默认值。"""
    return _algorithm_plugin(
        "agent",
        config_class,
        default_obs_agent_id=default_obs_agent_id,
    )


def learner_plugin(config_class: type) -> Callable[[type], type]:
    """联合声明 Learner 实现及 LearnerAlgoConfig 类型。"""
    return _algorithm_plugin("learner", config_class)


def algorithm_plugin_spec(implementation: type, kind: str) -> AlgorithmPluginSpec:
    """读取并校验算法插件声明。"""
    spec = getattr(implementation, "__dodo_algorithm_plugin__", None)
    if spec is None:
        raise TypeError(
            f"{kind} plugin {implementation.__module__}:{implementation.__name__} "
            f"does not declare its config class; use @{kind}_plugin(...)"
        )
    from config import AgentAlgoConfig, LearnerAlgoConfig

    expected = AgentAlgoConfig if kind == "agent" else LearnerAlgoConfig
    if not isinstance(spec.config_class, type) or not issubclass(spec.config_class, expected):
        name = getattr(spec.config_class, "__name__", repr(spec.config_class))
        raise TypeError(
            f"{kind} plugin {implementation.__module__}:{implementation.__name__} "
            f"config class {name} must inherit {expected.__name__}"
        )
    return spec


@dataclass
class PluginsConfig:
    """插件位置与启停策略。

    ``enabled`` 是白名单，``disabled`` 是黑名单；两者不能同时非空。
    两者皆空时采用黑名单策略，即当前清单内所有插件均启用。
    """

    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    locations: dict[str, dict[str, str]] = field(
        default_factory=lambda: {plugin_type: {} for plugin_type in PLUGIN_TYPES}
    )

    def copy(self) -> "PluginsConfig":
        return PluginsConfig(
            enabled=list(self.enabled),
            disabled=list(self.disabled),
            locations={plugin_type: dict(self.locations.get(plugin_type, {})) for plugin_type in PLUGIN_TYPES},
        )


@dataclass(frozen=True)
class GlobalPluginRegistry:
    """项目内置插件的全局定义。

    它不是 TOML 配置：五类项目插件在代码中统一维护。Portal 运行时以此为
    默认候选清单，训练 TOML 只负责选择禁用、白名单或追加的插件。
    """

    simulation: Mapping[str, str] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    agent: Mapping[str, str] = field(default_factory=dict)
    learner: Mapping[str, str] = field(default_factory=dict)
    module: Mapping[str, str] = field(default_factory=dict)

    def as_plugins_config(self) -> PluginsConfig:
        return PluginsConfig(
            locations={
                plugin_type: dict(getattr(self, plugin_type))
                for plugin_type in PLUGIN_TYPES
            }
        )


def _import_symbol(path: str) -> Any:
    module_name, separator, attr_name = path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(f"Invalid plugin entry point '{path}', expected module:function")
    # 安装版运行时全部位于 dodorl_sdk.dodorl；兼容旧 TOML 中 algos.* 等内置路径，
    # 但不创建任何顶层模块别名，避免与业务工程同名包冲突。
    if __package__.startswith("dodorl_sdk.dodorl."):
        root_name = module_name.partition(".")[0]
        if root_name in _BUNDLED_RUNTIME_ROOTS:
            module_name = f"dodorl_sdk.dodorl.{module_name}"
    return getattr(importlib.import_module(module_name), attr_name)


def _resolve_resource(target: Any) -> type:
    if isinstance(target, str):
        target = _import_symbol(target)
    if not isinstance(target, type):
        raise TypeError(f"Plugin resource must resolve to a class, given {target!r}")
    return target


class PluginRegistry:
    """一次运行会话的插件资源表。"""

    def __init__(self) -> None:
        self._resources: dict[str, dict[str, Any]] = {kind: {} for kind in RESOURCE_KINDS}
        self._sources: dict[str, dict[str, str]] = {kind: {} for kind in RESOURCE_KINDS}
        self._module_sources: dict[type, str] = {}
        self.active_plugins: list[str] = []
        self.disabled_plugins: list[str] = []

    def reset(self) -> None:
        for kind in RESOURCE_KINDS:
            self._resources[kind].clear()
            self._sources[kind].clear()
        self._module_sources.clear()
        self.active_plugins.clear()
        self.disabled_plugins.clear()
        clear_project_overrides()

    def _register_resource(self, kind: str, name: str, cls: type | str, plugin_id: str) -> None:
        if kind not in RESOURCE_KINDS:
            raise ValueError(f"Unsupported plugin resource kind: {kind}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Plugin '{plugin_id}' registered an empty {kind} name")
        previous = self._sources[kind].get(name)
        if previous is not None:
            raise PluginNameDuplicateException(
                plugin_type=kind,
                name=name,
                existing=previous,
                incoming=plugin_id,
            )
        self._resources[kind][name] = cls
        self._sources[kind][name] = plugin_id

    def activate(
            self,
            config: PluginsConfig,
            plugin_dirs: Iterable[str | Path] = (),
    ) -> None:
        self.reset()
        self._add_plugin_dirs(plugin_dirs)
        selectors = _select_plugins(config)
        self.disabled_plugins = list(config.disabled)
        for selector in selectors:
            plugin_type, separator, name = selector.partition(".")
            if not separator or not name or plugin_type not in PLUGIN_TYPES:
                raise ValueError(
                    f"Invalid plugin selector '{selector}', expected "
                    f"<type>.<name>, type in {sorted(PLUGIN_TYPES)}"
                )
            location = config.locations.get(plugin_type, {}).get(name)
            if location is None:
                if plugin_type == "learner" and name in RETIRED_LEARNERS:
                    raise ValueError(
                        f"Learner '{name}' was removed from DodoRL 0.17.0. "
                        "Migrate the training configuration or register a compatible custom learner plugin."
                    )
                raise ValueError(
                    f"plugins_config.enabled references undeclared plugin '{selector}'"
                )
            resource_kind = PLUGIN_TYPES[plugin_type]
            if resource_kind is None:
                self._register_module(name, _resolve_resource(location), selector)
            else:
                # 资源类按需导入，默认全启时不要求安装未使用场景的可选依赖。
                self._register_resource(resource_kind, name, location, selector)
            self.active_plugins.append(selector)
        if self.active_plugins:
            for selector in self.active_plugins:
                LOG.info(f"Enabled plugin: {selector}")
        else:
            LOG.info("Enabled plugins: (none)")

    def _register_module(self, name: str, custom_cls: type, source: str) -> None:
        bundled_classes = tuple(
            getattr(custom_cls, "__dodo_module_plugin_bundle__", ())
        )
        implementations = (custom_cls, *bundled_classes)
        overrides: dict[type, type] = {}
        for implementation in implementations:
            if not isinstance(implementation, type):
                raise TypeError(
                    f"Module plugin '{source}' bundle entries must be classes, got {implementation!r}"
                )
            base_cls = find_extensible_base(implementation)
            if base_cls is None:
                raise TypeError(
                    f"Module plugin '{source}' class {implementation.__name__} "
                    f"must inherit from an @extensible baseline class"
                )
            previous = self._module_sources.get(base_cls)
            if previous is not None:
                raise PluginImplementationDuplicateException(
                    baseline=base_cls.__name__,
                    existing=previous,
                    incoming=source,
                )
            if base_cls in overrides:
                raise PluginImplementationDuplicateException(
                    baseline=base_cls.__name__,
                    existing=source,
                    incoming=source,
                )
            overrides[base_cls] = implementation

        configure_project_overrides(overrides)
        for base_cls, implementation in overrides.items():
            self._module_sources[base_cls] = source
            LOG.debug(
                f"Module plugin override: {name}, "
                f"{base_cls.__name__} -> {implementation.__name__}"
            )

    @staticmethod
    def _add_plugin_dirs(plugin_dirs: Iterable[str | Path]) -> None:
        for raw_path in reversed(list(plugin_dirs)):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Plugin directory not found: {path}")
            path_string = str(path)
            if path_string not in sys.path:
                sys.path.insert(0, path_string)

    def resolve(self, kind: str, name: str) -> type:
        try:
            target = self._resources[kind][name]
        except KeyError as exc:
            if kind == "learner" and name in RETIRED_LEARNERS:
                raise ValueError(
                    f"Learner '{name}' was removed from DodoRL 0.17.0. "
                    "Migrate the training configuration or register a compatible custom learner plugin."
                ) from exc
            plugin_type = RESOURCE_PLUGIN_TYPES.get(kind, kind)
            selector = f"{plugin_type}.{name}"
            if selector in self.disabled_plugins:
                raise ValueError(
                    f"Plugin resource '{kind}:{name}' is explicitly disabled by "
                    f"[plugins_config].disabled ({selector})."
                ) from exc
            available = sorted(self._resources.get(kind, {}))
            raise ValueError(
                f"Unsupported {kind} plugin resource '{name}'. Available: {available}; "
                f"active plugins={self.active_plugins}"
            ) from exc
        return _resolve_resource(target)

    def has_resource(self, kind: str, name: str) -> bool:
        return name in self._resources.get(kind, {})

    def resource_names(self, kind: str) -> list[str]:
        return sorted(self._resources.get(kind, {}))

    def algorithm_spec(self, kind: str, name: str) -> AlgorithmPluginSpec:
        if kind not in {"agent", "learner"}:
            raise ValueError(f"Algorithm plugin kind must be agent or learner, got {kind!r}")
        return algorithm_plugin_spec(self.resolve(kind, name), kind)


RUNTIME_PLUGINS = PluginRegistry()


def _plugin_selectors(locations: dict[str, dict[str, str]]) -> list[str]:
    return [
        f"{plugin_type}.{name}"
        for plugin_type in PLUGIN_TYPES
        for name in locations.get(plugin_type, {})
    ]


def _validate_selector(selector: str) -> None:
    plugin_type, separator, name = selector.partition(".")
    if not separator or not name or plugin_type not in PLUGIN_TYPES:
        raise ValueError(
            f"Invalid plugin selector '{selector}', expected "
            f"<type>.<name>, type in {sorted(PLUGIN_TYPES)}"
        )


def _select_plugins(config: PluginsConfig) -> list[str]:
    if config.enabled and config.disabled:
        raise ValueError("[plugins_config].enabled and .disabled are mutually exclusive")

    declared = _plugin_selectors(config.locations)
    selected = config.enabled if config.enabled else declared
    excluded = set(config.disabled)
    for selector in [*config.enabled, *config.disabled]:
        _validate_selector(selector)
        if selector not in declared:
            kind, _, name = selector.partition(".")
            if kind == "learner" and name in RETIRED_LEARNERS:
                raise ValueError(
                    f"Learner '{name}' was removed from DodoRL 0.17.0. "
                    "Migrate the training configuration or register a compatible custom learner plugin."
                )
            raise ValueError(f"plugins_config references undeclared plugin '{selector}'")
    return [selector for selector in selected if selector not in excluded]


def merge_plugins_config(
        defaults: GlobalPluginRegistry | None,
        extra: PluginsConfig | None,
        *,
        enable_defaults: bool = False,
        allow_default_overrides: bool = False,
) -> PluginsConfig:
    """合并默认插件与 TOML 增量，并按调用方策略处理默认项和同名覆盖。"""
    merged = defaults.as_plugins_config() if defaults is not None else PluginsConfig()
    default_selectors = _plugin_selectors(merged.locations)
    if extra is None:
        if enable_defaults:
            merged.enabled = default_selectors
        return merged
    if extra.enabled and extra.disabled:
        raise ValueError("[plugins_config].enabled and .disabled are mutually exclusive")
    for plugin_type in PLUGIN_TYPES:
        target = merged.locations.setdefault(plugin_type, {})
        for name, location in extra.locations.get(plugin_type, {}).items():
            if name in target:
                existing = target[name]
                if not allow_default_overrides:
                    raise PluginNameDuplicateException(
                        plugin_type=plugin_type,
                        name=name,
                        existing=existing,
                        incoming=location,
                    )
                if _canonical_plugin_location(existing) == _canonical_plugin_location(location):
                    LOG.info(
                        f"Plugin declaration matches default: {plugin_type}.{name} -> {location}"
                    )
                    location = existing
                else:
                    LOG.warning(
                        f"Explicit plugin overrides default: {plugin_type}.{name}, "
                        f"default={existing}, explicit={location}"
                    )
            target[name] = location
    merged.enabled = (
        list(dict.fromkeys([*default_selectors, *extra.enabled]))
        if enable_defaults and extra.enabled
        else list(extra.enabled)
    )
    merged.disabled = list(extra.disabled)
    return merged


def activate_plugins(
        config: PluginsConfig | None = None,
        plugin_dirs: Iterable[str | Path] = (),
        *,
        default_plugins: GlobalPluginRegistry | None = None,
        enable_default_plugins: bool = False,
        allow_default_overrides: bool = False,
) -> PluginRegistry:
    RUNTIME_PLUGINS.activate(
        merge_plugins_config(
            default_plugins,
            config,
            enable_defaults=enable_default_plugins,
            allow_default_overrides=allow_default_overrides,
        ),
        plugin_dirs,
    )
    return RUNTIME_PLUGINS


def training_plugin_selectors(
        *,
        sense_name: str,
        agent_types: Iterable[str],
        algo_types: Iterable[str],
        sim_type: str | None = None,
        module_names: Iterable[str] = (),
) -> list[str]:
    """由训练配置推导应启用的插件选择器，顺序稳定、去重。"""
    selectors: list[str] = []
    if sim_type:
        selectors.append(f"simulation.{sim_type}")
    if sense_name:
        selectors.append(f"environment.{sense_name}")
    for agent_type in dict.fromkeys(agent_types):
        if agent_type:
            selectors.append(f"agent.{agent_type}")
    for algo_type in dict.fromkeys(algo_types):
        if algo_type:
            selectors.append(f"learner.{algo_type}")
    for module_name in dict.fromkeys(module_names):
        if module_name:
            selectors.append(f"module.{module_name}")
    return selectors


def bind_plugins_config_to_training(
        plugins_config: PluginsConfig,
        *,
        sense_name: str,
        agent_types: Iterable[str],
        algo_types: Iterable[str],
        sim_type: str | None = None,
        default_plugins: GlobalPluginRegistry | None = None,
) -> PluginsConfig:
    """
    TOML 未写 enabled/disabled 时，按科目实际引用收窄白名单。

    已显式白名单或黑名单时原样返回，不覆盖用户选择。
    module 插件仍全部纳入：它们会改写基线实现，与具体算法无关。
    """
    bound = plugins_config.copy()
    if bound.enabled or bound.disabled:
        return bound
    module_names: list[str] = []
    if default_plugins is not None:
        module_names.extend(default_plugins.module)
    module_names.extend(bound.locations.get("module", {}))
    bound.enabled = training_plugin_selectors(
        sense_name=sense_name,
        agent_types=agent_types,
        algo_types=algo_types,
        sim_type=sim_type,
        module_names=module_names,
    )
    return bound


__all__ = [
    "AlgorithmPluginSpec", "GlobalPluginRegistry", "PLUGIN_TYPES", "PluginRegistry",
    "PluginsConfig", "RUNTIME_PLUGINS", "activate_plugins", "agent_plugin",
    "algorithm_plugin_spec", "bind_plugins_config_to_training", "learner_plugin",
    "merge_plugins_config", "training_plugin_selectors",
]
