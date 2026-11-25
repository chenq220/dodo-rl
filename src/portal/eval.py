from __future__ import annotations

from ._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import initialize_role_runtime, load_training_config, parse_config_args
from common.portal_eval import resolve_eval_config, resolve_evaluation_class

LOG = init_logger("eval")

if __name__ == "__main__":
    conf = load_training_config(parse_config_args("策略评估"), PROJECT_ROOT, branch="eval")
    artifacts_manager = initialize_role_runtime(conf, role="eval", is_master=None)

    eval_config = resolve_eval_config(conf)
    evaluation_class = resolve_evaluation_class(conf)
    LOG.info(f"eval models: dir={eval_config.agent_model_path}, files={eval_config.agent_model_files}")
    evaluation_class(
        eval_config=eval_config,
        agent_config=conf.agents_config,
        env_config=conf.env_config,
        sim_config=conf.sim_config,
        sim_manager_config=conf.sim_manager_config,
        artifacts_manager=artifacts_manager,
    ).run()
