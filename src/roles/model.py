from __future__ import annotations

import time
from typing import Optional

from common.logger import init_logger
from config import GrpcConfig, ModelManagerConfig
from models import ModelManager
from roles.common.artifact_wiring import (
    bind_local_artifact_callers,
    bind_artifact_producer_callers,
    complete_artifact_producers,
    complete_role_logs,
    declare_role_log_producer,
)
from rpc.dispatch import ARTIFACTS_MANAGER, MODEL_MANAGER
from rpc.factory import RpcConfiguration, RpcServiceHost, create_rpc_client
from scheduler.artifact import ArtifactsManager
from utils.task_id import task_id

LOG = init_logger("model")


class ModelRole:
    """模型进程角色：持有ModelManager，并通过RPC提供模型版本和权重中转。"""

    def __init__(
            self,
            model_manager_config: ModelManagerConfig,
            model_manager_address: str,
            rpc_config: RpcConfiguration | None = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        self.model_manager_address = model_manager_address
        self.role_id = "model"
        self.rpc_config = rpc_config or GrpcConfig(model_manager_address=model_manager_address)
        self.model_manager = ModelManager(config=model_manager_config)
        self.artifacts_manager = artifacts_manager
        if artifacts_manager is not None:
            register, put, complete_for = bind_local_artifact_callers(
                self.artifacts_manager,
                lambda: str(task_id),
                lambda: "train",
            )
            bind_artifact_producer_callers(
                self.model_manager, "model_manager", register, put, complete_for,
            )
            self.model_manager.register_stage_intents()
        self._rpc_host = RpcServiceHost(
            self.rpc_config,
            "model-0",
            {MODEL_MANAGER: self.model_manager},
        )
        LOG.info("ModelRole initialized modules.")
        LOG.info(
            f"  ModelManager(tmp_local_dir={self.model_manager.config.tmp_local_dir}, "
            f"storage_type={self.model_manager.config.storage_type})"
        )

    def run(self) -> None:
        self._rpc_host.start()
        if self.artifacts_manager is None:
            self.artifacts_manager = create_rpc_client(self.rpc_config, ARTIFACTS_MANAGER)
        declare_role_log_producer(self.artifacts_manager, str(task_id), "train", self.role_id)
        LOG.info(f"ModelRole started, model_manager={self.model_manager_address}")
        try:
            while True:
                time.sleep(1.0)
        finally:
            try:
                # 收口导出模型池当前 champion 权重进台账，并上报 model_manager 报完。
                complete_artifact_producers(self.model_manager)
            except Exception:
                LOG.exception("ModelRole champion export failed during close")
            try:
                complete_role_logs(self.artifacts_manager, str(task_id), "train", self.role_id)
            except Exception:
                LOG.exception("ModelRole log ingest failed during close")
            self._rpc_host.stop()
            LOG.info(f"ModelRole finished: process run completed, task_id={task_id}.")
