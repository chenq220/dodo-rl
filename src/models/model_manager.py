from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, TYPE_CHECKING, Union

import torch
from torch import nn

from common.exception import ModelVersionNotFound, VarTypeErrorException
from scheduler.artifact import ArtifactCallerSupport, ArtifactType, ensure_output_parent
from common.logger import init_logger
from common.registry import extensible
from config import ModelManagerConfig, S3StorageConfig, StorageType
from models.model import ModelArtifactInfo, ModelInfo, ModelStatus, ModelUsage
from utils.task_id import task_id

LOG = init_logger("model_manager")

if TYPE_CHECKING:
    from common.types import AgentName, FilePath, ModelName


@extensible
class ModelManager(ArtifactCallerSupport):
    """
    模型池管理模块，负责当前模型寄存、查询和发布。

    设计边界：
    1. ModelManager 是模型生命周期管理器，不是参数平均器。
    2. 第一版不做模型评估、不做模型参数融合。
    3. 训练链路中 model_name 使用 AgentName，和 AgentConfig.name / LearnerConfig.name 对齐。
    4. 一个 model_name 保留 champion 和候选版本权重，保证候选模型不会覆盖当前发布模型。
    5. version 同时用于模型更新通知、运行期版本追踪和本地权重文件命名。
    6. 未来接 RPC 时，该类应部署在独立控制进程内，所有学习/采样进程通过 RPC 调用它。

    版本索引、champion 指针和自动晋升计数只在当前进程内有效；课程间模型继承必须通过
    model_load_dir + model_version 显式从 AutoML 归档读取，不能依赖本地工作目录遗留状态。
    """

    def __init__(self, config: ModelManagerConfig):
        # 模型池配置，描述本地临时权重目录、后端存储类型和首版自动晋升策略。
        self.config = config
        # 产物 register/put caller；由 Role 接线，未绑定时跳过。
        self._init_artifact_callers()
        # 本地版本自增计数，只负责生成版本号；每次新运行从零开始。
        self.update_count = 0
        # 本次 ModelManager 进程启动时间戳，用于生成可读版本号。
        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        # 任务 ID 是模型池文件名的第二段，用于在共享目录中区分不同训练任务的快照。
        self.task_id = self._safe_name(str(task_id))
        # 本地锁保证单进程内 submit/promote/rollback/register 的状态修改原子性。
        self._lock = threading.RLock()
        # 版本索引：model_name -> version -> ModelInfo。
        self._models: dict[str, dict[str, ModelInfo]] = {}
        # 当前发布版本索引：model_name -> champion version。
        self._champions: dict[str, str] = {}
        # 每个 Agent 已提交的有效学习批次数；自动晋升只在此处决策。
        self._learning_batches_completed: dict[str, int] = {}
        # 当前自动晋升周期中各 Agent 的候选版本；仅学习批次提交会进入该窗口。
        self._auto_promote_window_versions: dict[str, list[str]] = {}
        # 仅缓存本进程提交过的 nn.Module。
        self._module_cache: dict[tuple[str, str, str], nn.Module] = {}

    def register_stage_intents(self) -> None:
        self.register_artifact_intent(ArtifactType.MODEL)

    def complete_artifact_stage(self) -> None:
        """收口导出当前 champion 权重文件，再上报本模块本阶段产物报完。"""
        self._put_champion_artifacts()
        super().complete_artifact_stage()

    def _put_champion_artifacts(self) -> None:
        """归档前把模型池中各 model_name 当前 champion 的权重文件 put 进台账。

        候选版本、specialist、注册外部模型不 put；未接线 caller 时逐项跳过。
        """
        with self._lock:
            champions = [
                (model_name, self._models.get(model_name, {}).get(version))
                for model_name, version in self._champions.items()
            ]
        for model_name, model_info in champions:
            if model_info is None:
                continue
            for artifact_name, artifact_info in model_info.artifacts.items():
                self.put_artifact_file(
                    ArtifactType.MODEL,
                    artifact_info.local_path,
                    {
                        "model_name": str(model_name),
                        "champion_version": model_info.version,
                        "artifact": artifact_name,
                    },
                )

    @property
    def model_list(self) -> list[ModelInfo]:
        """兼容旧调用：返回当前模型池内全部版本记录。"""
        with self._lock:
            return [
                model_info
                for versions in self._models.values()
                for model_info in versions.values()
            ]

    def submit_model(
            self,
            model_name: AgentName,
            algo_type: str,
            models: Mapping[str, nn.Module | Mapping[str, Any]],
            *,
            parent_version: Optional[str] = None,
            learner_id: Optional[str] = None,
            train_steps: int = 0,
            metrics: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            version: Optional[str] = None,
            promote: bool = False,
            is_learning_batch: bool = False,
    ) -> ModelInfo:
        """
        提交一个候选模型版本。

        models 的 key 是权重名称，例如 agent_net、mixer_net；value 可以是 nn.Module 或 state_dict。
        is_learning_batch=True 表示本次提交来自一个完成的 Learner 学习批次。ModelManager
        以 Agent 为粒度累计批次，并依据 auto_promote_interval 自动晋升。开启
        auto_promote_optimo_reward 时，晋升当前周期 metrics 中 reward 最高的候选版本。
        """
        if not models:
            raise ValueError("models must not be empty")
        with self._lock:
            version = version or self._next_version(model_name)
            artifacts: dict[str, ModelArtifactInfo] = {}
            for artifact_name, model in models.items():
                safe_artifact_name = self._safe_name(artifact_name)
                file_name = self._artifact_file_name(model_name, version, safe_artifact_name)
                local_path = os.path.join(self.config.tmp_local_dir, file_name)
                ensure_output_parent(local_path)
                payload = self._model_payload(model)
                torch.save(payload, local_path)
                if isinstance(model, nn.Module):
                    self._module_cache[(model_name, version, artifact_name)] = model

                artifacts[artifact_name] = ModelArtifactInfo(
                    name=artifact_name,
                    usage=self._infer_usage(artifact_name).value,
                    local_path=local_path,
                    storage_key=file_name,
                )

            now = self._now()
            status = ModelStatus.CANDIDATE.value
            model_info = ModelInfo(
                model_name=model_name,
                version=version,
                algo_type=algo_type,
                status=status,
                parent_version=parent_version,
                artifacts=artifacts,
                metrics=metrics or {},
                metadata=metadata or {},
                history_versions=self._history_for(model_name),
                learner_id=learner_id,
                train_steps=train_steps,
                created_at=now,
                updated_at=now,
            )

            self._models.setdefault(model_name, {})[version] = model_info
            should_promote = promote or (
                    self.config.auto_promote_first_model and model_name not in self._champions
            )
            if is_learning_batch:
                batch_count = self._learning_batches_completed.get(model_name, 0) + 1
                self._learning_batches_completed[model_name] = batch_count
                interval = self.config.auto_promote_interval
                if interval > 0:
                    window = self._auto_promote_window_versions.setdefault(model_name, [])
                    window.append(version)
                    if batch_count % interval == 0:
                        version = self._resolve_auto_promote_version_unlocked(
                            model_name=model_name,
                            latest_version=version,
                            window_versions=window,
                        )
                        should_promote = True
                        self._auto_promote_window_versions[model_name] = []
            if should_promote:
                self._set_champion_unlocked(model_name, version)
            self._save_model_info(model_info)
            if version != model_info.version:
                # 评分最优版本可能早于本次提交；两份 metadata 都要反映最新状态。
                self._save_model_info(self._models[model_name][version])
            self._sync_to_backend(model_info)
            self._prune_model_versions_unlocked(model_name)
            LOG.info(
                f"Model submitted: model_name={model_name}, version={version}, "
                f"status={model_info.status}, artifacts={list(artifacts)}"
            )
            return model_info

    def _prune_model_versions_unlocked(self, model_name: AgentName) -> None:
        """回收模型池的过期本地快照，不影响显式保存的 final/optimo 文件。

        只处理由当前模型池目录管理的文件。注册的外部模型、champion、specialist、
        quarantined 版本，以及正在等待自动晋升决策的 candidates 都不会被回收。
        """
        max_versions = self.config.max_versions_per_model
        if max_versions == -1:
            return
        versions = self._models.get(model_name, {})
        if len(versions) <= max_versions:
            return

        protected_statuses = {
            ModelStatus.CHAMPION.value,
            ModelStatus.SPECIALIST.value,
            ModelStatus.QUARANTINED.value,
        }
        protected_versions = {
            version for version, info in versions.items()
            if info.status in protected_statuses or not self._is_local_managed_model(info)
        }
        protected_versions.update(self._auto_promote_window_versions.get(model_name, []))
        recent_versions = sorted(
            versions.values(), key=lambda info: (info.created_at, info.version), reverse=True,
        )[:max_versions]
        protected_versions.update(info.version for info in recent_versions)

        removable = [
            info for info in versions.values()
            if info.version not in protected_versions and self._is_local_managed_model(info)
        ]
        for info in removable:
            self._remove_local_version_files_unlocked(info)
            versions.pop(info.version, None)
            for artifact_name in info.artifacts:
                self._module_cache.pop((model_name, info.version, artifact_name), None)
        if removable:
            LOG.info(
                "Pruned local model snapshots: model_name=%s, removed=%d, retained=%d",
                model_name, len(removable), len(versions),
            )

    def _is_local_managed_model(self, model_info: ModelInfo) -> bool:
        """只有全部权重都位于模型池工作目录的版本才允许自动删除。"""
        pool_dir = os.path.realpath(self.config.tmp_local_dir)
        return bool(model_info.artifacts) and all(
            os.path.commonpath((pool_dir, os.path.realpath(artifact.local_path))) == pool_dir
            for artifact in model_info.artifacts.values()
        )

    def _remove_local_version_files_unlocked(self, model_info: ModelInfo) -> None:
        """删除一个已确认可回收版本的权重与单版本 metadata。"""
        paths = [artifact.local_path for artifact in model_info.artifacts.values()]
        paths.append(os.path.join(
            self.config.tmp_local_dir,
            self._metadata_file_name(model_info.model_name, model_info.version),
        ))
        for path in paths:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                LOG.warning("Failed to remove expired model snapshot: path=%s, error=%s", path, exc)

    def register_model(self,
                       model_info: Union[ModelInfo, Dict[str, Any], Iterable[ModelInfo | Dict[str, Any]]]) -> None:
        """
        注册已有模型版本记录，不保存权重文件。

        该接口用于从外部索引恢复、手动导入模型元数据，或者后续 RPC 同步其他 ModelManager
        服务端的版本记录。调用方必须保证 artifacts 中的路径或 storage_key 可用。
        """
        if isinstance(model_info, (ModelInfo, dict)):
            model_items = [model_info]
        else:
            model_items = list(model_info)
        with self._lock:
            for item in model_items:
                info = item if isinstance(item, ModelInfo) else ModelInfo.from_dict(item)
                self._models.setdefault(info.model_name, {})[info.version] = info
                if info.status == ModelStatus.CHAMPION.value:
                    self._champions[info.model_name] = info.version

    def get_model(
            self,
            model_name: AgentName,
            *,
            version: Optional[str] = None,
            output_type: Literal["info", "state_dict", "nn"] = "state_dict",
            artifact_names: Optional[Iterable[str]] = None,
    ) -> ModelInfo | Dict[str, Any] | Dict[str, nn.Module]:
        """
        获取模型版本信息、state_dict 或本进程缓存的 nn.Module。

        version=None 时优先返回 champion；如果尚无 champion，则返回该 model_name 下最新提交版本。
        output_type="nn" 只适合本进程内刚提交的模型，跨进程或重启后应使用 state_dict。
        """
        with self._lock:
            model_info = self._get_model_info_unlocked(model_name, version)
            if output_type == "info":
                return model_info
            artifacts = self._select_artifacts(model_info, artifact_names)
            if output_type == "state_dict":
                return {
                    artifact_name: torch.load(artifact.local_path, map_location="cpu")
                    for artifact_name, artifact in artifacts.items()
                }
            if output_type == "nn":
                modules = {}
                for artifact_name in artifacts:
                    cache_key = (model_name, model_info.version, artifact_name)
                    if cache_key not in self._module_cache:
                        raise ValueError(
                            "nn output is only available for modules submitted in the current ModelManager process. "
                            f"Missing cache: {cache_key}"
                        )
                    modules[artifact_name] = self._module_cache[cache_key]
                return modules
            raise ValueError(f"Unsupported output_type: {output_type}")

    @staticmethod
    def _select_artifacts(
            model_info: ModelInfo,
            artifact_names: Optional[Iterable[str]],
    ) -> Dict[str, ModelArtifactInfo]:
        """按调用方需要筛选模型产物，避免采样进程下载mixer、critic等训练专用网络。"""
        if artifact_names is None:
            return dict(model_info.artifacts)
        requested_names = list(dict.fromkeys(artifact_names))
        missing_names = [name for name in requested_names if name not in model_info.artifacts]
        if missing_names:
            raise KeyError(
                f"Model artifacts not found: model_name={model_info.model_name}, "
                f"version={model_info.version}, artifacts={missing_names}"
            )
        return {
            artifact_name: model_info.artifacts[artifact_name]
            for artifact_name in requested_names
        }

    def get_champion(self, model_name: AgentName) -> Optional[ModelInfo]:
        """返回当前发布版本；没有发布版本或 champion 指针悬空时返回 None。"""
        with self._lock:
            version = self._champions.get(model_name)
            if version is None:
                return None
            model_versions = self._models.get(model_name, {})
            if version not in model_versions:
                return None
            return model_versions[version]

    def promote_model(self, model_name: AgentName, version: Optional[str] = None) -> ModelInfo:
        """
        把指定版本晋升为 champion，旧 champion 自动归档。

        version=None 时晋升该 model_name 下最新提交版本（按 created_at / version 排序）。
        该接口供 Role、后续 ModelEvaluationWorker 或人工审核流程调用。晋升只改模型池状态，
        不修改任何权重文件。
        """
        with self._lock:
            target_version = version or self._resolve_latest_version_unlocked(model_name)
            self._require_version_unlocked(model_name, target_version)
            self._set_champion_unlocked(model_name, target_version)
            model_info = self._models[model_name][target_version]
            self._save_model_info(model_info)
            LOG.info(
                f"Model promoted: model_name={model_name}, version={target_version}"
            )
            return model_info

    def archive_model(self, model_name: AgentName, version: str) -> ModelInfo:
        """把指定版本标记为 archived，表示不发布但保留用于回滚、审计或继续训练。"""
        return self._set_status(model_name, version, ModelStatus.ARCHIVED)

    def quarantine_model(self, model_name: AgentName, version: str) -> ModelInfo:
        """把指定版本标记为 quarantined，表示模型存在硬错误，应避免采样或训练进程拉取。"""
        return self._set_status(model_name, version, ModelStatus.QUARANTINED)

    def mark_specialist(self, model_name: AgentName, version: str, tags: Optional[list[str]] = None) -> ModelInfo:
        """把指定版本标记为 specialist，并记录能力标签，例如 attack、defense。"""
        with self._lock:
            model_info = self._set_status_unlocked(model_name, version, ModelStatus.SPECIALIST)
            if tags:
                model_info.metadata["specialist_tags"] = tags
            self._save_model_info(model_info)
            return model_info

    def rollback_model(self, model_name: AgentName, version: Optional[str] = None) -> ModelInfo:
        """
        回滚到指定版本；未指定时回滚到当前 champion 的 parent_version。

        回滚本质上也是一次 promote，只是目标版本来自显式参数或当前 champion 的父版本。
        """
        with self._lock:
            current = self.get_champion(model_name)
            if current is None:
                raise ValueError(f"No champion model to rollback: model_name={model_name}")
            target_version = version or current.parent_version
            if target_version is None:
                raise ValueError(f"No rollback target for model: model_name={model_name}, champion={current.version}")
            return self.promote_model(model_name, target_version)

    def list_versions(
            self,
            model_name: Optional[AgentName] = None,
            status: Optional[str | ModelStatus] = None,
    ) -> list[ModelInfo]:
        """按模型名和状态筛选版本列表，用于模型池查询、审计和可视化。"""
        with self._lock:
            status_value = ModelStatus(status).value if status is not None else None
            if model_name is not None:
                versions = list(self._models.get(model_name, {}).values())
            else:
                versions = self.model_list
            if status_value is not None:
                versions = [info for info in versions if info.status == status_value]
            return sorted(versions, key=lambda info: (info.model_name, info.created_at, info.version))

    def get_agent_models(self) -> list[ModelInfo]:
        """返回包含 agent 权重的模型版本，供采样进程或模型展示界面筛选策略网络。"""
        with self._lock:
            return [
                info
                for info in self.model_list
                if any(artifact.usage == ModelUsage.AGENT.value for artifact in info.artifacts.values())
            ]

    def save_models(
            self,
            save_model_path: FilePath,
            save_model_name: Dict[ModelName, str],
            *,
            agent_name: AgentName,
            version: Optional[str] = None,
    ) -> list[str]:
        """
        将模型池中指定版本的权重按调用方给定的文件名落盘。

        TrainTaskManager 负责生成 save_model_name；本方法只 join 目录与 torch.save。
        """
        artifact_names = list(save_model_name.keys())
        state_dicts = self.get_model(
            agent_name,
            version=version,
            output_type="state_dict",
            artifact_names=artifact_names,
        )
        assert isinstance(state_dicts, dict)
        saved_files: list[str] = []
        for artifact_name, filename in save_model_name.items():
            if artifact_name not in state_dicts:
                raise KeyError(
                    f"Model artifact not found for save_models: model_name={agent_name}, "
                    f"artifact={artifact_name}"
                )
            file_path = os.path.join(save_model_path, filename)
            ensure_output_parent(file_path)
            torch.save(state_dicts[artifact_name], file_path)
            self.put_artifact_file(ArtifactType.MODEL, file_path)
            saved_files.append(file_path)
        LOG.info(
            f"ModelManager models saved: model_name={agent_name}, version={version}, "
            f"files={saved_files}"
        )
        return saved_files

    def _set_status(self, model_name: AgentName, version: str, status: ModelStatus) -> ModelInfo:
        """带锁更新版本状态，并更新该版本的 metadata。"""
        with self._lock:
            model_info = self._set_status_unlocked(model_name, version, status)
            self._save_model_info(model_info)
            return model_info

    def _set_status_unlocked(self, model_name: AgentName, version: str, status: ModelStatus) -> ModelInfo:
        """无锁状态更新；调用方必须已经持有 self._lock。"""
        model_info = self._require_version_unlocked(model_name, version)
        model_info.status = status.value
        model_info.updated_at = self._now()
        if self._champions.get(model_name) == version and status != ModelStatus.CHAMPION:
            self._champions.pop(model_name, None)
        return model_info

    def _set_champion_unlocked(self, model_name: AgentName, version: str) -> None:
        """无锁 champion 切换；调用方必须已经持有 self._lock。"""
        target = self._require_version_unlocked(model_name, version)
        old_version = self._champions.get(model_name)
        if old_version is not None and old_version != version:
            old_model = self._models.get(model_name, {}).get(old_version)
            if old_model is not None:
                old_model.status = ModelStatus.ARCHIVED.value
                old_model.updated_at = self._now()
                self._save_model_info(old_model)
                target.history_versions = old_model.history_versions + [old_version]
            if target.parent_version is None:
                target.parent_version = old_version
        target.status = ModelStatus.CHAMPION.value
        target.updated_at = self._now()
        self._champions[model_name] = version

    def _resolve_latest_version_unlocked(self, model_name: AgentName) -> str:
        """无锁解析 model_name 下最新提交版本。"""
        versions = self._models.get(model_name, {})
        if not versions:
            raise ValueError(f"Model not found: model_name={model_name}")
        return sorted(
            versions.values(),
            key=lambda info: (info.created_at, info.version),
        )[-1].version

    def _resolve_auto_promote_version_unlocked(
            self,
            model_name: AgentName,
            latest_version: str,
            window_versions: list[str],
    ) -> str:
        """在一个自动晋升周期内选择最后提交或 reward 最高的版本。"""
        if not self.config.auto_promote_optimo_reward:
            return latest_version
        scored = [
            (index, version, self._reward_metric_value(self._models[model_name][version]))
            for index, version in enumerate(window_versions)
            if version in self._models.get(model_name, {})
        ]
        scored = [candidate for candidate in scored if candidate[2] is not None]
        if not scored:
            LOG.warning(
                "auto_promote_optimo_reward is enabled but no numeric reward metric was submitted; "
                f"using latest model: model_name={model_name}, version={latest_version}"
            )
            return latest_version
        # 同分时取窗口内较晚的提交，符合未开启该选项时的默认行为。
        _, target_version, target_reward = max(scored, key=lambda item: (item[2], item[0]))
        LOG.info(
            "Auto-promotion selected optimo reward model: "
            f"model_name={model_name}, version={target_version}, reward={target_reward}"
        )
        return target_version

    @staticmethod
    def _reward_metric_value(model_info: ModelInfo) -> Optional[float]:
        """从 submit_model 的指标中解析 reward；兼容常见命名并忽略非有限值。"""
        metrics = model_info.metrics
        preferred_keys = (
            "reward",
            "episode_reward",
            "total_reward",
            "mean_reward",
            "reward_mean",
        )
        candidates = [metrics[key] for key in preferred_keys if key in metrics]
        candidates.extend(
            value
            for key, value in metrics.items()
            if "reward" in str(key).lower() and key not in preferred_keys
        )
        for value in candidates:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                return numeric
        return None

    def _get_model_info_unlocked(self, model_name: AgentName, version: Optional[str]) -> ModelInfo:
        """无锁获取版本记录；version 为空时按 champion/latest 规则解析。"""
        if version is None:
            version = self._champions.get(model_name)
            if version is None:
                version = self._resolve_latest_version_unlocked(model_name)
        return self._require_version_unlocked(model_name, version)

    def _require_version_unlocked(self, model_name: AgentName, version: str) -> ModelInfo:
        """无锁获取指定版本，不存在则立即报错。"""
        if model_name not in self._models or version not in self._models[model_name]:
            raise ModelVersionNotFound(f"Model version not found: model_name={model_name}, version={version}")
        return self._models[model_name][version]

    def _history_for(self, model_name: AgentName) -> list[str]:
        """生成新版本的历史链路快照。"""
        champion = self._champions.get(model_name)
        if champion is None:
            return []
        champion_versions = self._models.get(model_name, {})
        if champion not in champion_versions:
            return []
        champion_info = champion_versions[champion]
        return champion_info.history_versions + [champion]

    def _model_payload(self, model: nn.Module | Mapping[str, Any]) -> Mapping[str, Any]:
        """统一把 nn.Module 或 state_dict 转成可 torch.save 的权重字典。"""
        if isinstance(model, nn.Module):
            return model.state_dict()
        if isinstance(model, Mapping):
            return model
        raise VarTypeErrorException(name="model", given_var=model)

    def _next_version(self, model_name: AgentName) -> str:
        """生成本地递增版本号。"""
        self.update_count += 1
        return f"v{self.update_count:06d}_{self.timestamp}_{self._safe_name(model_name)}"

    def _artifact_file_name(self, model_name: AgentName, version: str, artifact_name: str) -> str:
        """返回包含 task_id 与版本号的扁平模型权重文件名。"""
        return "__".join([
            self._safe_name(model_name),
            self.task_id,
            self._safe_name(version),
            self._safe_name(artifact_name),
        ]) + ".pth"

    def _metadata_file_name(self, model_name: AgentName, version: str) -> str:
        """返回包含 task_id 与版本号的扁平模型元数据文件名。"""
        return "__".join([
            self._safe_name(model_name),
            self.task_id,
            self._safe_name(version),
            "metadata",
        ]) + ".json"

    def _save_model_info(self, model_info: ModelInfo) -> None:
        """写入单版本 metadata 文件；文件直接落在模型池根目录，不创建版本子目录。"""
        metadata_path = os.path.join(
            self.config.tmp_local_dir,
            self._metadata_file_name(model_info.model_name, model_info.version),
        )
        self._save_model_info_path = metadata_path
        ensure_output_parent(metadata_path)
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(model_info.to_dict(), fp, ensure_ascii=False, indent=2)

    def _sync_to_backend(self, model_info: ModelInfo) -> None:
        """把新提交版本的权重和 metadata 同步到 S3 协议后端。"""
        if self.config.storage_type != StorageType.S3.value:
            return
        storage = self.config.storage
        assert isinstance(storage, S3StorageConfig), (
            f"ModelManager S3 storage requires S3StorageConfig, given={type(storage).__name__}"
        )
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=storage.endpoint_url,
            aws_access_key_id=storage.access_key,
            aws_secret_access_key=storage.secret_key,
        )
        try:
            client.head_bucket(Bucket=storage.bucket_name)
        except client.exceptions.ClientError:
            client.create_bucket(Bucket=storage.bucket_name)

        for artifact in model_info.artifacts.values():
            client.upload_file(
                Filename=artifact.local_path,
                Bucket=storage.bucket_name,
                Key=artifact.storage_key,
            )
        metadata_key = self._metadata_file_name(model_info.model_name, model_info.version)
        metadata_path = os.path.join(self.config.tmp_local_dir, metadata_key)
        client.upload_file(
            Filename=metadata_path,
            Bucket=storage.bucket_name,
            Key=metadata_key,
        )

    @staticmethod
    def _infer_usage(artifact_name: str) -> ModelUsage:
        """根据权重名称推断用途，调用方后续也可以通过 metadata 精细标注。"""
        lowered = artifact_name.lower()
        if "agent" in lowered or "policy" in lowered:
            return ModelUsage.AGENT
        if "mixer" in lowered or "mix" in lowered:
            return ModelUsage.MIXER
        if "critic" in lowered or "value" in lowered:
            return ModelUsage.CRITIC
        if "learner" in lowered:
            return ModelUsage.LEARNER
        return ModelUsage.OTHER

    @staticmethod
    def _safe_name(value: str) -> str:
        """把模型名、版本名转换成适合文件路径和对象 key 的安全字符串。"""
        safe_value = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value)).strip("_")
        return safe_value or "model"

    @staticmethod
    def _now() -> str:
        """生成模型元数据中的时间字符串。"""
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
