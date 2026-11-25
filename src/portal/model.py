from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from config import ModelManagerConfig
from roles.model import ModelRole

LOG = init_logger("model")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("分布式模型服务"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="model", is_master=False)

    model_manager_config = conf.model_manager_config or ModelManagerConfig()
    ModelRole(
        model_manager_config=model_manager_config,
        model_manager_address=getattr(conf.rpc_config, "model_manager_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
