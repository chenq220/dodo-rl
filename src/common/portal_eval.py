from __future__ import annotations

import copy
import importlib
from typing import Any, Callable, Type

from config import AgentConfig, EvalConfig, SampleType
from roles.eval_control import EvalControlRole
from roles.evaluation import EvaluationRole
from utils.toml_config import TomlTrainingConfig


def _import_symbol(path: str) -> Any:
    module_name, _, attr_name = path.partition(":")
    if not attr_name:
        raise ValueError(f"Invalid import path (expected module:attr): {path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def resolve_eval_config(conf: TomlTrainingConfig) -> EvalConfig:
    builder_path = conf.entry_config.eval_config_builder
    if builder_path:
        builder: Callable[[TomlTrainingConfig], EvalConfig] = _import_symbol(builder_path)
        eval_config = builder(conf)
        if not isinstance(eval_config, EvalConfig):
            raise TypeError(
                f"{builder_path} must return EvalConfig, got {type(eval_config).__name__}"
            )
        return eval_config
    return conf.eval_config


def resolve_evaluation_class(conf: TomlTrainingConfig) -> Type[EvaluationRole]:
    class_path = conf.entry_config.evaluation_class
    if class_path:
        evaluation_class = _import_symbol(class_path)
        if not issubclass(evaluation_class, EvaluationRole):
            raise TypeError(f"{class_path} must be a subclass of EvaluationRole")
        return evaluation_class
    return EvaluationRole


def resolve_distributed_evaluation_class(conf: TomlTrainingConfig) -> Type[EvalControlRole]:
    class_path = conf.entry_config.distributed_evaluation_class
    if class_path:
        evaluation_class = _import_symbol(class_path)
        if not issubclass(evaluation_class, EvalControlRole):
            raise TypeError(f"{class_path} must be a subclass of EvalControlRole")
        return evaluation_class
    return EvalControlRole


def apply_eval_model_file_paths(
        eval_config: EvalConfig,
        agent_configs: list[AgentConfig],
) -> list[AgentConfig]:
    """把 EvalConfig 中的评估模型文件写入各 AgentConfig.model_file_path。"""
    configs = [copy.deepcopy(cfg) for cfg in agent_configs]
    model_file_paths = EvaluationRole._resolve_model_file_paths(eval_config, configs)
    for cfg in configs:
        if cfg.name in model_file_paths:
            cfg.model_file_path = model_file_paths[cfg.name]
    return configs


def bind_eval_config_for_sample(eval_config: EvalConfig) -> EvalConfig:
    """把评估配置绑定为 SampleRole 使用的在线采样契约。"""
    eval_config.sample_type = SampleType.ONLINE.value
    return eval_config
