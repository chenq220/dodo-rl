"""Optuna 隔离适配层。

整个 dodorl 框架内**唯一**允许 `import optuna` 的模块。scheduler/automl.py、
train_task.py、Role 等都只经 scheduler.utils.optuna_adapter 与 Optuna 交互。

升级 Optuna 版本时，只需修改本文件 + requirements.txt 的版本上界；上层模块无需改动。
当前基线：optuna>=4.0.0,<5.0.0（Ask-and-Tell 公开 API）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.trial import TrialState

from common.logger import init_logger
from config import AutoMLConfig, AutoMLOptunaConfig, LearnerConfig
from scheduler.model import AutoMLParamRangeConfig, ParamLayer, resolve_tunable_param_enum

if TYPE_CHECKING:
    from scheduler.model import TunableParamSpec


@dataclass
class OptunaTrialHandle:
    """框架内 trial 句柄；不暴露 optuna.Trial。"""
    trial_number: int


@dataclass
class OptunaPruneDecision:
    should_prune: bool
    reason: str = ""


@dataclass
class OptunaStudySummary:
    """写入 advice.yaml 的 optuna 段；纯 dict 亦可。"""
    trial_number: int
    state: str
    params: Dict[str, Any]
    value: Optional[float]
    best_trial_number: Optional[int]
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]


def configure_optuna_project_logging() -> None:
    """将 Optuna 日志纳入项目的 ``common.logger`` handler 链路。

    Optuna 默认会给 ``optuna`` logger 安装独立的 stderr handler，输出形如
    ``[I ...] A new study created ...``。移除该 handler 后，以同名 logger 注册到项目
    日志系统，控制台、文件、格式和 task_id 都跟随当前 ``[log_config]``。

    本函数可重复调用：``init_logger`` 会复用同名 logger，适合每个 study 打开前调用。
    """
    optuna.logging.disable_default_handler()
    init_logger("optuna")


def _spec_to_distribution(spec: "TunableParamSpec", override: Optional[AutoMLParamRangeConfig]) -> BaseDistribution:
    """把 TunableParamSpec（可被 TOML override 收窄）映射为 Optuna distribution。"""
    low = override.low if override and override.low is not None else spec.low
    high = override.high if override and override.high is not None else spec.high
    log = override.log if override and override.log is not None else spec.log
    step = override.step if override and override.step is not None else spec.step
    choices = override.choices if override and override.choices is not None else spec.choices

    if spec.dist_type == "float":
        if low is None or high is None:
            raise ValueError(f"float TunableParam '{spec.key}' requires low/high, got low={low}, high={high}")
        # Optuna 不允许 log=True 时带 step；log 模式强制 step=None。
        float_step = None if log else (step or 0.0)
        return FloatDistribution(low=float(low), high=float(high), log=bool(log), step=float_step)
    if spec.dist_type == "int":
        if low is None or high is None:
            raise ValueError(f"int TunableParam '{spec.key}' requires low/high, got low={low}, high={high}")
        return IntDistribution(low=int(low), high=int(high), step=int(step or 1), log=bool(log))
    if spec.dist_type == "categorical":
        if choices is None:
            raise ValueError(f"categorical TunableParam '{spec.key}' requires choices")
        return CategoricalDistribution(choices=tuple(choices))
    raise ValueError(f"unsupported dist_type='{spec.dist_type}' for TunableParam '{spec.key}'")


def _learner_spec_path(spec: "TunableParamSpec") -> str:
    return f"algo_config.{spec.key}" if spec.layer == ParamLayer.ALGO_CONFIG else spec.key


def _free_distribution(path: str, dist_type: str, config: AutoMLParamRangeConfig) -> BaseDistribution:
    if dist_type == "float":
        if config.low is None or config.high is None:
            raise ValueError(f"float env metadata parameter {path!r} requires low/high")
        if config.log and config.step is not None:
            raise ValueError(f"float env metadata parameter {path!r}: log=true cannot be combined with step")
        return FloatDistribution(
            low=float(config.low), high=float(config.high), log=bool(config.log),
            step=None if config.log else (float(config.step) if config.step is not None else None),
        )
    if dist_type == "int":
        if config.low is None or config.high is None:
            raise ValueError(f"int env metadata parameter {path!r} requires low/high")
        if config.log and config.step is not None:
            raise ValueError(f"int env metadata parameter {path!r}: log=true cannot be combined with step")
        return IntDistribution(
            low=int(config.low), high=int(config.high), log=bool(config.log),
            step=int(config.step or 1),
        )
    if dist_type == "categorical":
        choices = config.choices
        if not choices:
            raise ValueError(f"categorical env metadata parameter {path!r} requires non-empty choices")
        if any(not isinstance(v, (type(None), bool, int, float, str)) for v in choices):
            raise ValueError(f"categorical env metadata parameter {path!r} has unsupported choices")
        if len({(type(v), v) for v in choices}) != len(choices):
            raise ValueError(f"categorical env metadata parameter {path!r} has duplicate choices")
        return CategoricalDistribution(choices=tuple(choices))
    raise ValueError(f"unsupported env metadata distribution type={dist_type!r} for path={path!r}")


def build_study_distributions(
        learner_configs: List[LearnerConfig],
        config: AutoMLConfig,
) -> Dict[str, BaseDistribution]:
    """将 Learner、环境 metadata 与种子三类入口合并为单个 Optuna trial。"""
    merged: Dict[str, BaseDistribution] = {}
    learners_by_name = {item.name: item for item in learner_configs}
    for item in config.learner_search_space:
        learner = learners_by_name.get(item.agent_name)
        if learner is None:
            raise ValueError(
                f"learner_search_space references unknown agent_name={item.agent_name!r}; "
                f"available={sorted(learners_by_name)}"
            )
        enum_cls = resolve_tunable_param_enum(learner.algo_type)
        spec = next((m.value for m in enum_cls if _learner_spec_path(m.value) == item.path), None)
        if spec is None:
            available = sorted(_learner_spec_path(m.value) for m in enum_cls)
            raise ValueError(
                f"Learner {item.agent_name!r} path={item.path!r} is not tunable; available={available}"
            )
        key = f"learner.{item.agent_name}.{item.path}"
        if key in merged:
            raise ValueError(f"duplicate AutoML search parameter {key!r}")
        merged[key] = _spec_to_distribution(spec, item)

    for item in config.env_metadata_search_space:
        if not item.path or any(not part for part in item.path.split(".")):
            raise ValueError(f"invalid env metadata search path={item.path!r}")
        key = f"env_metadata.{item.path}"
        if key in merged:
            raise ValueError(f"duplicate AutoML search parameter {key!r}")
        merged[key] = _free_distribution(item.path, item.type, item)

    if config.seed_search is not None:
        low, high = int(config.seed_search.low), int(config.seed_search.high)
        if not 0 <= low <= high <= 2147483647:
            raise ValueError(f"seed_search requires 0 <= low <= high <= 2147483647, got {low}..{high}")
        merged["seed"] = IntDistribution(low=low, high=high)

    if config.optuna.sampler_name == "cmaes" and any(
        isinstance(distribution, CategoricalDistribution) for distribution in merged.values()
    ):
        raise ValueError("CMA-ES does not support categorical AutoML parameters; use tpe or random")
    return merged


def create_sampler(config: AutoMLOptunaConfig):
    name = config.sampler_name
    if name == "tpe":
        return optuna.samplers.TPESampler(n_startup_trials=config.sampler_n_startup_trials)
    if name == "random":
        return optuna.samplers.RandomSampler()
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(n_startup_trials=config.sampler_n_startup_trials)
    raise ValueError(f"unsupported AutoMLSamplerType name='{name}'")


def create_pruner(config: AutoMLOptunaConfig):
    name = config.pruner_name
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=config.pruner_n_startup_trials,
            n_warmup_steps=config.pruner_n_warmup_steps,
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=config.pruner_min_resource,
            max_resource=config.pruner_max_resource,
            reduction_factor=config.pruner_reduction_factor,
        )
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"unsupported AutoMLPrunerType name='{name}'")


class OptunaStudySession:
    """Ask-and-Tell 会话；AutoML 持有一个实例。"""

    def __init__(self, study: optuna.Study):
        self._study = study
        # trial_number → optuna.trial.Trial；report/tell 需要原始 trial 句柄。
        self._trials: Dict[int, Any] = {}

    @classmethod
    def open(
            cls,
            config: AutoMLOptunaConfig,
            *,
            study_name: str,
            storage: str,
    ) -> "OptunaStudySession":
        # Optuna 只能在本适配层被导入/配置，日志也在这里一次性接入项目日志体系。
        configure_optuna_project_logging()
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=config.direction,
            sampler=create_sampler(config),
            pruner=create_pruner(config),
            load_if_exists=config.load_if_exists,
        )
        return cls(study)

    def ask(self, distributions: Dict[str, BaseDistribution]) -> Tuple[OptunaTrialHandle, Dict[str, Any]]:
        trial = self._study.ask(fixed_distributions=distributions)
        self._trials[trial.number] = trial
        return OptunaTrialHandle(trial_number=trial.number), dict(trial.params)

    def report(self, handle: OptunaTrialHandle, value: float, step: int) -> OptunaPruneDecision:
        trial = self._trials[handle.trial_number]
        trial.report(float(value), step)
        should_prune = trial.should_prune()
        reason = "optuna_should_prune" if should_prune else ""
        return OptunaPruneDecision(should_prune=should_prune, reason=reason)

    def tell_pruned(self, handle: OptunaTrialHandle) -> None:
        trial = self._trials.pop(handle.trial_number, None)
        if trial is None:
            self._study.tell(handle.trial_number, state=TrialState.PRUNED)
            return
        self._study.tell(trial, state=TrialState.PRUNED)

    def tell_complete(self, handle: OptunaTrialHandle, value: float) -> None:
        trial = self._trials.pop(handle.trial_number, None)
        if trial is None:
            self._study.tell(handle.trial_number, values=[float(value)])
            return
        self._study.tell(trial, values=[float(value)])

    def export_summary(self, handle: Optional[OptunaTrialHandle] = None) -> OptunaStudySummary:
        trials = self._study.trials
        # handle=None 表示当前运行是未占用 Optuna trial 的基线配置；不能把 Study 中上一条
        # 历史 trial 误写成当前运行。best_trial 仍在下方独立导出。
        current_number = handle.trial_number if handle else -1
        current = next((t for t in trials if t.number == current_number), None)
        best_trial_number = None
        best_value = None
        best_params = None
        try:
            best = self._study.best_trial
            best_trial_number = best.number
            best_value = best.value
            best_params = dict(best.params)
        except ValueError:
            # 还没有 COMPLETE 的 trial，best_trial 不可用。
            pass
        return OptunaStudySummary(
            trial_number=current_number,
            state=current.state.name if current is not None else "NONE",
            params=dict(current.params) if current is not None else {},
            value=current.value if current is not None else None,
            best_trial_number=best_trial_number,
            best_value=best_value,
            best_params=best_params,
        )
