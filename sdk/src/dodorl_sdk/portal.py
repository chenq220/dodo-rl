"""面向使用者项目的训练、采样与评估角色 SDK 入口。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable




from importlib import import_module
from .environment import bootstrap_environment

_RUNTIME_IMPORTS = {'PluginRegistry': 'common.plugins:PluginRegistry',
 'PluginsConfig': 'common.plugins:PluginsConfig',
 'activate_plugins': 'common.plugins:activate_plugins',
 'initialize_role_runtime': 'common.portal_entry:initialize_role_runtime',
 'apply_eval_model_file_paths': 'common.portal_eval:apply_eval_model_file_paths',
 'bind_eval_config_for_sample': 'common.portal_eval:bind_eval_config_for_sample',
 'resolve_distributed_evaluation_class': 'common.portal_eval:resolve_distributed_evaluation_class',
 'resolve_eval_config': 'common.portal_eval:resolve_eval_config',
 'resolve_evaluation_class': 'common.portal_eval:resolve_evaluation_class',
 'ModelManagerConfig': 'config:ModelManagerConfig',
 'MonteCarloMethodConfig': 'config:MonteCarloMethodConfig',
 'ControlRole': 'roles.control:ControlRole',
 'LearnRole': 'roles.learn:LearnRole',
 'LearnControlRole': 'roles.learn_control:LearnControlRole',
 'ModelRole': 'roles.model:ModelRole',
 'SampleRole': 'roles.sample:SampleRole',
 'SerialStageTrainRole': 'roles.serial_stage_train:SerialStageTrainRole',
 'SerialTrainRole': 'roles.serial_train:SerialTrainRole',
 'StageControlRole': 'roles.stage_control:StageControlRole',
 'StageLearnRole': 'roles.stage_learn_control:StageLearnRole',
 'TomlTrainingConfig': 'utils.toml_config:TomlTrainingConfig',
 'task_id': 'utils.task_id:task_id',
 'get_project_root': '.config:get_project_root',
 'SDK_DEFAULT_PLUGIN_REGISTRY': '.plugins:SDK_DEFAULT_PLUGIN_REGISTRY'}
_RUNTIME_LOADED = False


def _load_runtime():
    global _RUNTIME_LOADED
    if _RUNTIME_LOADED:
        return
    for name, path in _RUNTIME_IMPORTS.items():
        module, attribute = path.split(":", 1)
        globals()[name] = getattr(import_module(module, __package__), attribute)
    _RUNTIME_LOADED = True


def __getattr__(name):
    if name in _RUNTIME_IMPORTS:
        _load_runtime()
        return globals()[name]
    raise AttributeError(name)


class SDKPluginRegistry:
    """SDK 插件会话：加载用户插件并校验角色需要的环境资源。"""

    def __init__(self, plugin_dirs: Iterable[str | Path] | str | Path = ()) -> None:
        if isinstance(plugin_dirs, (str, Path)):
            plugin_dirs = (plugin_dirs,)
        self.plugin_dirs = tuple(Path(path).expanduser().resolve() for path in plugin_dirs)
        self.runtime: PluginRegistry | None = None

    def activate(self, config: PluginsConfig | TomlTrainingConfig) -> PluginRegistry:
        _load_runtime()
        plugins_config = config.plugins_config if isinstance(config, TomlTrainingConfig) else config
        self.runtime = activate_plugins(
            plugins_config,
            self.plugin_dirs,
            default_plugins=SDK_DEFAULT_PLUGIN_REGISTRY,
            enable_default_plugins=True,
            allow_default_overrides=True,
        )
        return self.runtime

    def require_environment(self, sense_name: str) -> None:
        if self.runtime is None:
            raise RuntimeError("SDK plugins must be activated before resource validation")
        if not self.runtime.has_resource("env", sense_name):
            raise ValueError(
                f"SDK requires an enabled environment plugin for '{sense_name}'. "
                f"Available environments: {self.runtime.resource_names('env')}; "
                f"active plugins={self.runtime.active_plugins}"
            )


class _BaseRoleSDK:
    """从统一 TOML 路径或文本构造角色，也可显式覆盖仿真配置对象。"""

    branch = "train"
    role_name = ""
    is_master: bool | None = None
    default_role_id_suffix: str | None = None

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            runtime_role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        if (config_file is None) == (config_text is None):
            raise ValueError("Provide exactly one of config_file or config_text")
        self.config_file = Path(config_file).expanduser().resolve() if config_file is not None else None
        if self.config_file is not None and not self.config_file.is_file():
            raise FileNotFoundError(f"SDK training TOML not found: {self.config_file}")
        bootstrap_environment(self.config_file, config_text=config_text)
        _load_runtime()
        self.plugins = SDKPluginRegistry(plugin_dirs)
        # 先轻量配置 task_id / logger，再激活插件，避免激活日志早于 logger。
        plugins_config = TomlTrainingConfig.prepare_runtime_bootstrap(
            self.config_file,
            project_root=get_project_root(),
            config_text=config_text,
        )
        self.plugins.activate(plugins_config)
        self.config = TomlTrainingConfig(
            self.config_file,
            project_root=get_project_root(),
            config_text=config_text,
            plugin_registry=self.plugins.runtime,
            branch=self.branch,
        )
        # 未显式覆盖时保留统一 TOML 中委托 dodo-sim 装配的配置。
        if sim_config is not None:
            self.config.sim_config = sim_config
        if sim_manager_config is not None:
            self.config.sim_manager_config = sim_manager_config
        self.plugins.require_environment(self.config.env_config.sense_name)
        resolved_role_id = runtime_role_id
        if self.default_role_id_suffix is not None:
            resolved_role_id = runtime_role_id or self._resolve_role_id(
                self.default_role_id_suffix,
            )
            self.role_id = resolved_role_id
        self.artifacts_manager = initialize_role_runtime(
            self.config,
            role=self.role_name,
            is_master=self.is_master,
            role_id=resolved_role_id,
            activate_portal_plugins=False,
        )

    def _resolve_role_id(self, default_suffix: str) -> str:
        """按 Portal 相同规则解析角色 ID。"""
        if self.config.entry_config.role_id:
            return self.config.entry_config.role_id
        task_name = self.config.design.get("task_name") or self.config.design.get("sense_name") or "task"
        return f"{task_name}-{default_suffix}"


class ControlSDK(_BaseRoleSDK):
    """运行独立控制角色，承载任务、回放、仿真资源和中心样本录制服务。"""

    role_name = "control"
    is_master = True

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        ControlRole(
            train_config=self.config.train_config,
            replay_buffer_config=self.config.replay_buffer_config,
            policy_agent_configs=self.config.step_agents_config,
            learner_configs=self.config.learners_config,
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            sim_manager_config=self.config.sim_manager_config,
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_config=(
                self.config.monte_carlo_method_config or MonteCarloMethodConfig()
            ),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class StageControlSDK(_BaseRoleSDK):
    """运行完整分布式拓扑的独立分阶段任务控制角色。"""

    role_name = "stage_control"
    is_master = True
    default_role_id_suffix = "stage-control-0"

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        super().__init__(plugin_dirs, config_file, runtime_role_id=role_id,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        StageControlRole(
            role_id=self.role_id,
            stage_task_config=self.config.stage_task_config,
            env_config=self.config.env_config,
            replay_buffer_config=self.config.replay_buffer_config,
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            sim_manager_config=self.config.sim_manager_config,
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_config=(
                self.config.monte_carlo_method_config or MonteCarloMethodConfig()
            ),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class LearnSDK(_BaseRoleSDK):
    """运行连接独立控制与模型服务的学习角色。"""

    role_name = "learn"
    is_master = False
    default_role_id_suffix = "learn-0"

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        super().__init__(plugin_dirs, config_file, runtime_role_id=role_id,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        LearnRole(
            role_id=self.role_id,
            train_config=self.config.train_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            step_agents_config=self.config.step_agents_config,
            learners_config=self.config.learners_config,
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class ModelSDK(_BaseRoleSDK):
    """运行独立模型版本与权重发布服务。"""

    role_name = "model"
    is_master = False

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        ModelRole(
            model_manager_config=self.config.model_manager_config or ModelManagerConfig(),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class LearnControlSDK(_BaseRoleSDK):
    """运行分布式训练的中心学习控制角色。"""

    role_name = "learn_control"
    is_master = True
    default_role_id_suffix = "learn-control-0"

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        super().__init__(plugin_dirs, config_file, runtime_role_id=role_id,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        LearnControlRole(
            role_id=self.role_id,
            train_config=self.config.train_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            step_agents_config=self.config.step_agents_config,
            learners_config=self.config.learners_config,
            replay_buffer_config=self.config.replay_buffer_config,
            model_manager_config=self.config.model_manager_config or ModelManagerConfig(),
            sim_manager_config=self.config.sim_manager_config,
            monte_carlo_method_config=(
                self.config.monte_carlo_method_config or MonteCarloMethodConfig()
            ),
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class StageLearnSDK(_BaseRoleSDK):
    """运行分布式分阶段训练的中心学习控制角色。"""

    role_name = "stage_learn"
    is_master = True
    default_role_id_suffix = "learn-0"

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        super().__init__(plugin_dirs, config_file, runtime_role_id=role_id,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        StageLearnRole(
            role_id=self.role_id,
            stage_task_config=self.config.stage_task_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            replay_buffer_config=self.config.replay_buffer_config,
            model_manager_config=self.config.model_manager_config or ModelManagerConfig(),
            sim_manager_config=self.config.sim_manager_config,
            monte_carlo_method_config=(
                self.config.monte_carlo_method_config or MonteCarloMethodConfig()
            ),
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class SampleSDK(_BaseRoleSDK):
    """运行一个连接分布式中心服务的采样角色。评估时传 ``branch='eval'``，仍启动 SampleRole。"""

    role_name = "sample"
    is_master = False

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            sample_role_name: str,
            branch: str = "train",
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        if not sample_role_name.strip():
            raise ValueError("sample_role_name must not be empty")
        self.branch = str(branch or "train").strip() or "train"
        self.sample_role_name = sample_role_name
        super().__init__(plugin_dirs, config_file, runtime_role_id=sample_role_name,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        if self.branch == "eval":
            train_config = bind_eval_config_for_sample(resolve_eval_config(self.config))
            step_agents_config = apply_eval_model_file_paths(
                train_config,
                self.config.step_agents_config,
            )
            learners_config: list = []
        else:
            train_config = self.config.train_config
            step_agents_config = self.config.step_agents_config
            learners_config = self.config.learners_config
        SampleRole(
            role_id=self.sample_role_name,
            train_config=train_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            step_agents_config=step_agents_config,
            learners_config=learners_config,
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
        ).run()


class SerialStageTrainSDK(_BaseRoleSDK):
    """以用户插件目录与训练 TOML 直接运行串行分阶段训练。"""

    role_name = "serial_stage_train"
    is_master = None

    def run(self) -> None:
        sim_manager_config = self.config.sim_manager_config
        SerialStageTrainRole(
            stage_task_config=self.config.stage_task_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            replay_buffer_config=self.config.replay_buffer_config,
            model_manager_config=self.config.model_manager_config,
            sim_manager_config=sim_manager_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class SerialTrainSDK(_BaseRoleSDK):
    """运行单进程、单阶段的串行训练角色。"""

    role_name = "serial_train"
    is_master = None

    def run(self) -> None:
        sim_manager_config = self.config.sim_manager_config
        SerialTrainRole(
            config=self.config.train_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            replay_buffer_config=self.config.replay_buffer_config,
            step_agents_config=self.config.step_agents_config,
            learners_config=self.config.learners_config,
            sim_manager_config=sim_manager_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class EvaluationSDK(_BaseRoleSDK):
    """以同一插件集与 TOML 运行策略评估。"""

    branch = "eval"
    role_name = "eval"
    is_master = None

    def run(self) -> None:
        eval_config = resolve_eval_config(self.config)
        evaluation_class = resolve_evaluation_class(self.config)
        evaluation_class(
            eval_config=eval_config,
            agent_config=self.config.agents_config,
            env_config=self.config.env_config,
            sim_config=self.config.sim_config,
            sim_manager_config=self.config.sim_manager_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


class EvalControlSDK(_BaseRoleSDK):
    """运行分布式评估的中心控制角色。"""

    branch = "eval"
    role_name = "eval_control"
    is_master = True
    default_role_id_suffix = "eval-control-0"

    def __init__(
            self,
            plugin_dirs: Iterable[str | Path] | str | Path = (),
            config_file: str | Path | None = None,
            *,
            role_id: str | None = None,
            config_text: str | None = None,
            sim_config=None,
            sim_manager_config=None,
    ) -> None:
        super().__init__(plugin_dirs, config_file, runtime_role_id=role_id,
                         config_text=config_text, sim_config=sim_config, sim_manager_config=sim_manager_config)

    def run(self) -> None:
        rpc_config = self.config.rpc_config
        evaluation_class = resolve_distributed_evaluation_class(self.config)
        evaluation_class(
            role_id=self.role_id,
            eval_config=resolve_eval_config(self.config),
            env_config=self.config.env_config,
            step_agents_config=self.config.step_agents_config,
            replay_buffer_config=self.config.replay_buffer_config,
            model_manager_config=self.config.model_manager_config or ModelManagerConfig(),
            sim_manager_config=self.config.sim_manager_config,
            monte_carlo_method_config=(
                self.config.monte_carlo_method_config or MonteCarloMethodConfig()
            ),
            task_manager_address=getattr(rpc_config, "task_manager_address", ""),
            replay_buffer_address=getattr(rpc_config, "replay_buffer_address", ""),
            model_manager_address=getattr(rpc_config, "model_manager_address", ""),
            sim_manager_address=getattr(rpc_config, "sim_manager_address", ""),
            monte_carlo_method_address=getattr(rpc_config, "monte_carlo_method_address", ""),
            rpc_config=rpc_config,
            artifacts_manager=self.artifacts_manager,
        ).run()


__all__ = [
    "SDKPluginRegistry",
    "ControlSDK",
    "StageControlSDK",
    "LearnSDK",
    "ModelSDK",
    "LearnControlSDK",
    "StageLearnSDK",
    "SampleSDK",
    "SerialStageTrainSDK",
    "SerialTrainSDK",
    "EvaluationSDK",
    "EvalControlSDK",
]
