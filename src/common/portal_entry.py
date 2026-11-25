from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

from common.info_bridge import INFO_IS_MASTER, info_bridge, publish_training_info
from common.logger import log_startup_process
from common.plugins import activate_plugins, bind_plugins_config_to_training
from scheduler.artifact import ArtifactsManager, build_artifacts_manager
from utils.task_id import task_id
from utils.toml_config import TomlTrainingConfig


def resolve_config_file(config_file: str | Path, project_root: Path) -> Path:
    """解析 -f 路径：支持 cwd、仓库根、subjects/ 三种相对基准。"""
    raw = Path(config_file).expanduser()
    if raw.is_absolute():
        if not raw.is_file():
            raise FileNotFoundError(f"Config file not found: {raw}")
        return raw.resolve()

    subjects_dir = project_root / "subjects"
    candidates = [
        Path.cwd() / raw,
        project_root / raw,
        subjects_dir / raw,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Config file not found: {config_file}. Tried: {tried}"
    )


def parse_config_args(
        description: str,
        *,
        extra_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    if extra_arguments is not None:
        extra_arguments(parser)
    parser.add_argument(
        "-f",
        "--config-file",
        required=True,
        help=(
            "TOML 配置文件路径；相对路径依次按当前目录、仓库根、subjects/ 解析，"
            "例如 subjects/cooperative_search/cooperative_search_qmix.toml 或 cooperative_search/cooperative_search_qmix.toml。"
        ),
    )
    return parser.parse_args()


def load_training_config(
        args: argparse.Namespace,
        project_root: Path,
        *,
        branch: str = "train",
) -> TomlTrainingConfig:
    """纯配置加载：路径解析并构造 TomlTrainingConfig，不做插件激活与运行时副作用。"""
    config_path = resolve_config_file(args.config_file, project_root)
    # TOML 中 {{project_root}} 等运行路径以启动 working directory 为准；
    # project_root 参数仍仅用于定位仓库内置配置文件。
    return TomlTrainingConfig(
        config_file=config_path,
        project_root=Path.cwd().resolve(),
        branch=branch,
    )


def _activate_portal_plugins(conf: TomlTrainingConfig) -> None:
    """项目 Portal：按科目引用收窄白名单后激活全局注册表。"""
    from portal.plugins.registry import project_plugin_registry

    defaults = project_plugin_registry()
    agent_types = [agent.agent_type for agent in conf.agents_config]
    algo_types = [learner.algo_type for learner in conf.learners_config]
    for stage in conf.stage_task_config.stages:
        agent_types.extend(agent.agent_type for agent in stage.agents_config)
        algo_types.extend(learner.algo_type for learner in stage.learners_config)
    algo_types.extend(
        learner.algo_type for learner in conf.stage_task_config.bootstrap_learners_config
    )
    plugins_config = bind_plugins_config_to_training(
        conf.plugins_config,
        sense_name=conf.env_config.sense_name,
        agent_types=agent_types,
        algo_types=algo_types,
        sim_type=conf.sim_config.sim_type if conf.env_config.use_sim else None,
        default_plugins=defaults,
    )
    activate_plugins(plugins_config, default_plugins=defaults)


def initialize_role_runtime(
        conf: TomlTrainingConfig,
        *,
        role: str,
        is_master: bool | None,
        role_id: str | None = None,
        activate_portal_plugins: bool = True,
) -> ArtifactsManager | None:
    """
    角色运行时初始化：声明 IS_MASTER、可选 Portal 插件激活、发布配置、
    freeze / Startup process。中心/串行构建并激活 ArtifactsManager；worker 不激活第二份权威 AM。
    """
    info_bridge.put(**{INFO_IS_MASTER: is_master})
    if activate_portal_plugins:
        _activate_portal_plugins(conf)
    publish_training_info(conf)
    task_id.freeze()
    log_startup_process(role, conf.config_file, role_id)
    if is_master is False:
        return None
    if conf.branch == "eval":
        save_dir = conf.eval_config.artifacts_archive_save_dir
    elif conf.stage_task_config.stages:
        save_dir = conf.stage_task_config.stages[0].train_config.artifacts_archive_save_dir
    else:
        save_dir = conf.train_config.artifacts_archive_save_dir
    manager = build_artifacts_manager(
        artifacts_manager_config=conf.artifacts_manager_config,
        artifacts_archive_save_dir=save_dir,
        raw_file=conf.raw_file if (is_master is True or is_master is None) else None,
        config_name=str(conf.config_file) if (is_master is True or is_master is None) else None,
        activate=True,
    )
    runtime_dir = conf.design.get("artifacts_dir")
    # 旧配置的 artifacts_dir 可能是跨任务共享根；只绑定明确按任务命名的目录。
    if runtime_dir and Path(runtime_dir).name == str(task_id):
        manager.bind_task_runtime_directory(str(task_id), runtime_dir)
    return manager


def resolve_role_id(conf: TomlTrainingConfig, default_suffix: str) -> str:
    entry = conf.entry_config
    if entry.role_id:
        return entry.role_id
    task_name = conf.design.get("task_name") or conf.design.get("sense_name") or "task"
    return f"{task_name}-{default_suffix}"
