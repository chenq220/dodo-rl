from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from roles.control import ControlRole

LOG = init_logger("control")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("分布式中心控制"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="control", is_master=True)

    ControlRole(
        train_config=conf.train_config,
        replay_buffer_config=conf.replay_buffer_config,
        policy_agent_configs=conf.step_agents_config,
        learner_configs=conf.learners_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        sim_manager_config=conf.sim_manager_config,
        sim_manager_address=getattr(conf.rpc_config, "sim_manager_address", ""),
        monte_carlo_method_config=conf.monte_carlo_method_config,
        monte_carlo_method_address=getattr(conf.rpc_config, "monte_carlo_method_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
