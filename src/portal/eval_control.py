from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimManagerConfig


try:
    from portal._path import bootstrap_sys_path
except ModuleNotFoundError:
    from ._path import bootstrap_sys_path

PROJECT_ROOT = bootstrap_sys_path()

from common.logger import init_logger
from common.portal_entry import (
    initialize_role_runtime,
    load_training_config,
    parse_config_args,
    resolve_role_id,
)
from common.portal_eval import resolve_distributed_evaluation_class, resolve_eval_config
from config import ModelManagerConfig, MonteCarloMethodConfig

LOG = init_logger("eval_control")


def _parse_args():
    def add_role_id(parser):
        parser.add_argument(
            "--role-id",
            default=None,
            help="进程角色 ID，默认从 TOML 推导。",
        )

    return parse_config_args("分布式评估控制", extra_arguments=add_role_id)


if __name__ == "__main__":
    args = _parse_args()
    conf = load_training_config(args, PROJECT_ROOT, branch="eval")
    role_id = args.role_id or resolve_role_id(conf, "eval-control-0")
    artifacts_manager = initialize_role_runtime(
        conf, role="eval_control", is_master=True, role_id=role_id,
    )

    eval_config = resolve_eval_config(conf)
    evaluation_class = resolve_distributed_evaluation_class(conf)
    model_manager_config = conf.model_manager_config or ModelManagerConfig()
    monte_carlo_method_config = conf.monte_carlo_method_config or MonteCarloMethodConfig()
    sim_manager_config = conf.sim_manager_config
    evaluation_class(
        role_id=role_id,
        eval_config=eval_config,
        env_config=conf.env_config,
        step_agents_config=conf.step_agents_config,
        replay_buffer_config=conf.replay_buffer_config,
        model_manager_config=model_manager_config,
        sim_manager_config=sim_manager_config,
        monte_carlo_method_config=monte_carlo_method_config,
        task_manager_address=getattr(conf.rpc_config, "task_manager_address", ""),
        replay_buffer_address=getattr(conf.rpc_config, "replay_buffer_address", ""),
        model_manager_address=getattr(conf.rpc_config, "model_manager_address", ""),
        sim_manager_address=getattr(conf.rpc_config, "sim_manager_address", ""),
        monte_carlo_method_address=getattr(conf.rpc_config, "monte_carlo_method_address", ""),
        rpc_config=conf.rpc_config,
        artifacts_manager=artifacts_manager,
    ).run()
