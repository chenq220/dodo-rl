# 串行单阶段训练入口
from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from roles.serial_train import SerialTrainRole

LOG = init_logger("serial_train")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("单进程串行训练"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="serial_train", is_master=None)

    SerialTrainRole(
        config=conf.train_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        replay_buffer_config=conf.replay_buffer_config,
        step_agents_config=conf.step_agents_config,
        learners_config=conf.learners_config,
        sim_manager_config=conf.sim_manager_config,
        artifacts_manager=artifacts_manager,
    ).run()
