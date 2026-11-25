"""AutoML 门面：对外稳定契约为 AutoML.advice() / archive_train() / on_stage_complete()。

实现细节分布在 scheduler.utils：
- automl_core：指标、坍塌、参数拆分、路径命名
- automl_config：配置 finalize
- automl_archive_storage：归档落盘与 S3
- optuna_adapter：唯一 import optuna 的模块
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional

from common.logger import init_logger
from common.registry import extensible
from config import AutoMLConfig, LearnerConfig
from scheduler.model import (
    AutoMLAdvice,
    TrainingResourceContext,
    TrainingResourceContribution,
    TrainingResourceEvent,
    TrainingRunArchive,
    TrainingSnapshot,
)
from scheduler.utils.automl_archive_storage import (
    persist_archive_bundle,
    write_archive_bundle_contents,
)
from scheduler.utils.automl_core import (
    TrialPeakTracker,
    aggregate_reward,
    build_archive_folder_name,
    build_restart_advice,
    check_in_trial_collapse,
    check_target_metric_deadline,
    resolve_optuna_sqlite_storage,
)
from scheduler.utils.optuna_adapter import (
    OptunaStudySession,
    OptunaTrialHandle,
    build_study_distributions,
)

LOG = init_logger("automl")


def _params_for_log(params: Dict[str, object]) -> tuple[dict, dict, object]:
    learner = {key: value for key, value in params.items() if key.startswith("learner.")}
    env = {key: value for key, value in params.items() if key.startswith("env_metadata.")}
    return learner, env, params.get("seed")

# 测试与少量调用方仍从此处 re-export
__all__ = [
    "AutoML",
    "TrialPeakTracker",
    "aggregate_reward",
    "build_archive_folder_name",
    "build_restart_advice",
    "check_in_trial_collapse",
    "check_target_metric_deadline",
    "resolve_optuna_sqlite_storage",
]


@extensible
class AutoML:
    def __init__(
            self,
            config: AutoMLConfig,
            learner_configs: List[LearnerConfig],
            *,
            stage_name: str,
    ):
        self.config = config
        self._learner_configs = learner_configs
        study_key, storage = resolve_optuna_sqlite_storage(config.subject_name, stage_name)
        self._session = OptunaStudySession.open(
            config.optuna,
            study_name=study_key,
            storage=storage,
        )
        self._distributions = build_study_distributions(learner_configs, config)
        self._trial: Optional[OptunaTrialHandle] = None
        self._trial_params: Dict[str, object] = {}
        self._archive_current_trial: Dict[str, object] = {}
        self._peak_tracker = TrialPeakTracker()
        self._extra_complete_rounds_remaining = max(0, int(config.extra_complete_rounds))
        LOG.info(
            f"AutoML initialized study={study_key} with baseline config; "
            f"Optuna ask is deferred until a restart is required"
        )

    def _restart_after_prune(self, reason: str, metric_value: float) -> AutoMLAdvice:
        """结束当前 trial（若有）并为下一次运行申请一组新参数。"""
        if self._trial is not None:
            self._archive_current_trial = {
                "number": self._trial.trial_number,
                "state": "PRUNED",
                "value": metric_value,
                "params": dict(self._trial_params),
            }
            self._session.tell_pruned(self._trial)
        else:
            self._archive_current_trial = {
                "number": None, "state": "BASELINE", "value": metric_value, "params": {},
            }
        self._trial, self._trial_params = self._session.ask(self._distributions)
        self._peak_tracker = TrialPeakTracker()
        learner_params, env_params, seed = _params_for_log(self._trial_params)
        LOG.info(
            f"AutoML asked restart trial={self._trial.trial_number}, "
            f"learner_params={learner_params}, env_params={env_params}, "
            f"seed={seed}, reason={reason}"
        )
        return build_restart_advice(
            self._trial_params,
            self._learner_configs,
            reason=reason,
        )

    def advice(self, snapshot: TrainingSnapshot) -> AutoMLAdvice:
        if not self._distributions:
            return AutoMLAdvice(decision="continue", reason="automl_no_search_space")

        metric_value = aggregate_reward(snapshot, self.config)
        decision = None
        if self._trial is not None:
            decision = self._session.report(
                self._trial,
                metric_value,
                snapshot.completed_episodes,
            )
        collapse_reason = check_in_trial_collapse(
            metric_value,
            snapshot.completed_episodes,
            self.config,
            self._peak_tracker,
        )
        if collapse_reason:
            LOG.info(
                f"AutoML advice prune (collapse) at episode={snapshot.completed_episodes}, "
                f"trial={getattr(self._trial, 'trial_number', 'baseline')}, metric={metric_value}, "
                f"peak={self._peak_tracker.peak}, reason={collapse_reason}, "
                f"generation={snapshot.generation}"
            )
            return self._restart_after_prune(collapse_reason, metric_value)

        target_reason = check_target_metric_deadline(
            metric_value,
            snapshot.completed_episodes,
            self.config,
        )
        if target_reason:
            LOG.info(
                f"AutoML advice prune (target missed) at episode={snapshot.completed_episodes}, "
                f"trial={getattr(self._trial, 'trial_number', 'baseline')}, metric={metric_value}, reason={target_reason}, "
                f"generation={snapshot.generation}"
            )
            return self._restart_after_prune(target_reason, metric_value)

        if self._trial is None:
            LOG.info(
                f"AutoML baseline advice continue at episode={snapshot.completed_episodes}, "
                f"metric={metric_value}, peak={self._peak_tracker.peak}"
            )
            return AutoMLAdvice(decision="continue")

        if decision is not None and not decision.should_prune:
            LOG.info(
                f"AutoML advice continue at episode={snapshot.completed_episodes}, "
                f"trial={self._trial.trial_number}, metric={metric_value}, "
                f"peak={self._peak_tracker.peak}, generation={snapshot.generation}"
            )
            return AutoMLAdvice(decision="continue")

        LOG.info(
            f"AutoML advice prune at episode={snapshot.completed_episodes}, "
            f"trial={self._trial.trial_number}, metric={metric_value}, reason={decision.reason}"
        )
        return self._restart_after_prune(decision.reason or "optuna_pruned", metric_value)

    def on_sample_completed(self, snapshot: TrainingSnapshot) -> Optional[AutoMLAdvice]:
        """采样跑满一轮：tell(COMPLETE)；extra_complete_rounds 未用尽则 ask 下一 trial 并返回 restart 建议。"""
        if self._trial is not None:
            metric_value = aggregate_reward(snapshot, self.config)
            trial_number = self._trial.trial_number
            self._archive_current_trial = {
                "number": trial_number, "state": "COMPLETE", "value": metric_value,
                "params": dict(self._trial_params),
            }
            self._session.tell_complete(self._trial, metric_value)
            self._trial = None
            LOG.info(
                f"AutoML tell COMPLETE on sample completed: trial={trial_number}, "
                f"metric={metric_value}, episodes={snapshot.completed_episodes}, "
                f"extra_rounds_remaining={self._extra_complete_rounds_remaining}"
            )
        else:
            self._archive_current_trial = {
                "number": None, "state": "BASELINE",
                "value": aggregate_reward(snapshot, self.config), "params": {},
            }
        if self._extra_complete_rounds_remaining <= 0 or not self._distributions:
            return None
        self._extra_complete_rounds_remaining -= 1
        self._trial, self._trial_params = self._session.ask(self._distributions)
        self._peak_tracker = TrialPeakTracker()
        learner_params, env_params, seed = _params_for_log(self._trial_params)
        LOG.info(
            f"AutoML extra complete round: new_trial={self._trial.trial_number}, "
            f"learner_params={learner_params}, env_params={env_params}, seed={seed}, "
            f"extra_rounds_remaining={self._extra_complete_rounds_remaining}"
        )
        return build_restart_advice(
            self._trial_params,
            self._learner_configs,
            reason="extra_complete_round",
        )

    def training_resource_callback(
            self,
            context: TrainingResourceContext,
    ) -> Optional[TrainingResourceContribution]:
        """TrainTaskManager 资源生命周期回调；监听归档资源就绪与重启完成。"""
        if context.event == TrainingResourceEvent.PRE_ARCHIVE_REQUEST:
            return None
        if context.event == TrainingResourceEvent.ARCHIVE_RESOURCES_READY:
            LOG.info(
                f"AutoML archive resources ready: stage={context.stage_name}, "
                f"generation={context.generation}, episodes={context.completed_episodes}, "
                f"reason={context.reason}"
            )
            return None
        if context.event == TrainingResourceEvent.POST_RESTART:
            LOG.info(
                f"AutoML post-restart settled: stage={context.stage_name}, "
                f"generation={context.generation}, prior_episodes={context.completed_episodes}, "
                f"reason={context.reason}"
            )
            return None
        return None

    def on_stage_complete(self, snapshot: TrainingSnapshot) -> None:
        """阶段正常结束（非 prune）时 tell COMPLETE；采样跑满时通常已在 on_sample_completed 收口。"""
        if self._trial is None:
            self._archive_current_trial = {
                "number": None, "state": "BASELINE", "value": aggregate_reward(snapshot, self.config),
                "params": {},
            }
            return
        metric_value = aggregate_reward(snapshot, self.config)
        LOG.info(
            f"AutoML tell COMPLETE at stage end: trial={self._trial.trial_number}, "
            f"metric={metric_value}, episodes={snapshot.completed_episodes}, "
            f"generation={snapshot.generation}"
        )
        self._session.tell_complete(self._trial, metric_value)
        self._archive_current_trial = {
            "number": self._trial.trial_number, "state": "COMPLETE", "value": metric_value,
            "params": dict(self._trial_params),
        }
        self._trial = None

    def archive_train(self, bundle: TrainingRunArchive) -> str:
        """代次结束时落盘归档包；覆盖 prune、extra round reset 与正常完成。"""
        if not self.config.archive_enabled:
            LOG.info(
                f"AutoML archive skipped (archive_enabled=false): "
                f"stage={bundle.stage_name}, generation={bundle.generation}"
            )
            return ""
        archive_cfg = self.config
        folder_name = build_archive_folder_name(bundle)
        archive_dir = os.path.join(archive_cfg.archive_root_dir, folder_name)
        os.makedirs(archive_dir, exist_ok=True)
        exported = asdict(self._session.export_summary(self._trial))
        metric = self.config.business_metric
        if metric is not None and metric.enabled:
            metric_name = metric.name
            aggregation = metric.aggregation
            window = metric.window
            target = metric.target
            deadline = metric.deadline_episodes
        else:
            metric_name = self.config.metric_primary
            aggregation = "mean"
            window = self.config.reward_window
            target = self.config.target_metric
            deadline = self.config.target_metric_deadline_episodes
        metric_value = aggregate_reward(bundle.snapshot, self.config)
        direction = self.config.optuna.direction
        target_achieved = None if target is None else (
            metric_value >= target if direction == "maximize" else metric_value <= target
        )
        optuna_payload = {
            "study_name": self.config.subject_name + "_" + bundle.stage_name,
            "current_trial": self._archive_current_trial or {
                "number": None, "state": "BASELINE", "value": metric_value, "params": {},
            },
            "next_trial": ({
                "number": exported["trial_number"],
                "state": exported["state"],
                "params": exported["params"],
            } if self._trial is not None else None),
            "best_trial": {
                "number": exported["best_trial_number"],
                "value": exported["best_value"],
                "params": exported["best_params"] or {},
            },
            "metric": {
                "name": metric_name,
                "aggregation": aggregation,
                "window": window,
                "value": metric_value,
                "direction": direction,
                "target": target,
                "deadline_episodes": deadline,
                "target_achieved": target_achieved,
            },
        }
        write_archive_bundle_contents(bundle, archive_dir, optuna_payload)
        archive_location = persist_archive_bundle(archive_cfg, archive_dir, folder_name)
        LOG.info(f"AutoML archived run to {archive_location}")
        return archive_location
