from __future__ import annotations

from portal._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from roles.random_sampling import RandomSamplingRole

LOG = init_logger("random_sampling")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("随机策略采样"), PROJECT_ROOT)
    artifacts_manager = initialize_role_runtime(conf, role="random_sampling", is_master=None)

    RandomSamplingRole(
        monte_carlo_method_config=conf.monte_carlo_method_config,
        agent_config=conf.agents_config[0],
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        artifacts_manager=artifacts_manager,
    ).run()
