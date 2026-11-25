# 分阶段串行训练入口
from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from roles.serial_stage_train import SerialStageTrainRole

LOG = init_logger("serial_stage_train")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("单进程分阶段训练"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="serial_stage_train", is_master=None)

    SerialStageTrainRole(
        stage_task_config=conf.stage_task_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        replay_buffer_config=conf.replay_buffer_config,
        model_manager_config=conf.model_manager_config,
        sim_manager_config=conf.sim_manager_config,
        artifacts_manager=artifacts_manager,
    ).run()
